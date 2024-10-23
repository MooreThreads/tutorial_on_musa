#!/bin/bash
# Date: 2024-10-30
# Author: create by GPU-Genius
# Version: 1.0
# Function: It is used to view the hardware and software information of the current machine host environment, including CPU manufacturer, GPU number, driver version, etc., and provides necessary environment inspection tips for developers. You can get the information above directly from your terminal by executing: bash ./musa_report.sh



# Function to display error message and exit
function error_exit {
    echo "Error: $1"
    exit 1
}

WARNING_LOG=()
GMI_FLAG=1
PCIE_VERSION=''
MTLINK_STATUS="disabled"

# define color codes for ANSI escape sequences
RED='\033[0;31m'  
GREEN='\033[0;32m' 
NC='\033[0m' # No Color, used to reset colors


function pcie_version() {
    first_pci_address=`lspci -n | grep 1ed5 | head -n 1 | awk '{print $1}'`
    
    if [ -z "$first_pci_address" ]; then
        echo -e "${RED}Error: output of \" lspci -n | grep 1ed5 | head -n 1 | awk '{print $1}'\" is empty, please check you device!${NC}"
        exit 1
    fi

    pci_status=`sudo lspci -vvvvs ${first_pci_address} |grep Lnk`
    speed=$(echo "${pci_status}" | grep "LnkSta:" | grep -oP '\d+GT/s' | head -n 1 | grep -oP '\d+')

    # ------------------------------------------
    # Speed 32GT/s = 2^5 GT/s, means "PCIE Gen5"
    # Speed 16GT/s = 2^4 GT/s, means "PCIE Gen4"
    # Speed 8GT/s = 2^3 GT/s, means "PCIE Gen3"
    # ------------------------------------------
    if [ -z "$speed" ]; then
        echo -e "${RED}Warning: Speed value not found in lspci log.${NC}"
    elif [ "$speed" -eq 32 ]; then
        PCIE_VERSION='Gen5'
    elif [ "$speed" -eq 16 ]; then
        PCIE_VERSION='Gen4'
    elif [ "$speed" -eq 8 ]; then
        PCIE_VERSION='Gen3'
    else
        echo -e "${RED}Warning: Speed is ${speed},  but is unkown PCIE version!${NC}"
    fi
}

function check_host_env() {
    # 1. Check the operating system version
    SystemVersion=$(lsb_release -d | sed 's/^Description:\s*//')

    # 2. Check the kernel version
    # - 5.4.xx-5.15.xxx
    KernelVersion=$(uname -r)
    # echo $KernelVersion
    # if ! [[ "$kernel_version" =~ ^5\.([4-9]|1[0-5])\. ]]; then
    #     error_exit "Unsupported kernel version. Please use kernel version between 5.4.xx and 5.15.xx."
    # fi
    
    # 3. Check IOMMU
    if grep -qE "intel_iommu=on|amd_iommu=on" /etc/default/grub; then
	IOMMU_STATUS="${GREEN}enabled${NC}"
    else
        IOMMU_STATUS="${RED}disabled${NC}"
        IOMMU_WARNING="IOMMU should be enable, but now is disabled, please refer to xxxx.md to enable IOMMU!"
        WARNING_LOG+=("$IOMMU_WARNING")
    fi 

    # 4. Check GPU numbers by cmd lscpi
    GPUDevices=$(lspci -n| grep -i '1ed5' | wc -l)
    ((GPUDevices = GPUDevices / 2))
    if [ $GPUDevices -eq 0 ]; then
        GPU_WARNING="Unable to get pcie device information, please check if your hardware is installed correctly!"
        WARNING_LOG+=("$GPU_WARNING")
    fi

    # Install clinfo 
    if ! command -v clinfo &> /dev/null;then
        echo "command \"clinfo\" is not found, attempting to install clinfo"
        if ! (sudo apt update -y &>/dev/null && \
            sudo apt --fix-broken install -y &>/dev/null && \
            sudo apt install -y clinfo &>/dev/null); then
            CLINFO_WARNING="Failed to install clinfo, can not get Driver Version through clinfo command, please check your machine environment or network environment!"
	    WARNING_LOG+=("$CLINFO_WARNING")
        fi
    fi
    
    # 5. Check mthreads-gmi version
    GMI_VERSION=$(mthreads-gmi -v 2> /dev/null | awk -F: '{print $2}' | tr -d ' ')
    if [ -z "$GMI_VERSION" ]; then
        GMI_VERSION="N/A"
        DRIVER_WARNING="mthrads-gmi: command not found, unable to get mthreads-gmi version, MTBios version..., please check if your DDK is installed correctly!"
        WARNING_LOG+=("$DRIVER_WARNING")
        GMI_FLAG=0
    fi

    # 6. Check MTBios Version
    if [ $GMI_FLAG -eq 0 ]; then
        MTBios_VERSION="N/A"
    else
        MTBios_VERSION=$(mthreads-gmi -q | grep "MTBios" | awk -F ': ' '{print $2}' | tr -d ' ')
    fi

    # 7. Check pcie Version
    # 检查脚本是否有sudo权限
    if [ "$EUID" -ne 0 ]
    then
        PCIE_VERSION="N/A"
        PCIE_VERSION_WARNING="PCIE version is not found, please try to execute the script using \"${RED}sudo${NC} bash $0\" or as ${RED}root${NC}!"
        WARNING_LOG+=("$PCIE_VERSION_WARNING")
    else
        pcie_version
        if [ -z $PCIE_VERSION ]; then
            PCIE_VERSION="N/A"
            PCIE_VERSION_WARNING="PCIE version is not found, please check it!"
            WARNING_LOG+=("$PCIE_VERSION_WARNING")
        fi
    fi

    # 8. Check the driver verison
    if [ $GMI_FLAG -eq 0 ]; then
        DRIVER_VERSION_GMI="N/A"
        DRIVER_VERSION_CLINFO="N/A"
    else
        DRIVER_VERSION_GMI=$(mthreads-gmi | grep "Driver Version" | awk -F 'Driver Version:' '{print $2}')
        DRIVER_VERSION_CLINFO=$(clinfo | grep "Driver Version" -m 1 | awk '{print $3, $5}' | sed 's/ / /')
    fi
    if [ -z "$DRIVER_VERSION_CLINFO" ]; then
        DRIVER_VERSION_CLINFO="N/A"
    fi

    # 9. Check mt-container-toolkit
    mt_container_toolkit_version=$(dpkg-query -W 2> /dev/null | grep mt-container-toolkit | awk '{print $2}')
    if [ -z "$mt_container_toolkit_version" ]; then
        mt_container_toolkit_version="N/A"
    fi
    docker run --rm --env MTHREADS_VISIBLE_DEVICES=all registry.mthreads.com/cloud-mirror/ubuntu:20.04 mthreads-gmi &> /dev/null
    if [ $? -ne 0 ]; then
        MT_CONTAINER_TOOLKIT_STATUS="${RED}failed${NC}"
        MT_CONTAINER_TOOLKIT_WARNING="Running 'sudo docker run --rm --env MTHREADS_VISIBLE_DEVICES=all registry.mthreads.com/cloud-mirror/ubuntu:20.04 mthreads-gmi' to check MT-Container-Toolkit Status failed, please refer to https://docs.mthreads.com/cloud-native/cloud-native-doc-online/install_guide"
        WARNING_LOG+=("$MT_CONTAINER_TOOLKIT_WARNING")
    else
        MT_CONTAINER_TOOLKIT_STATUS="${GREEN}successful${NC}"
    fi


    # 10. Check the Docker 
    DOCKER_VERSION=$(docker --version 2> /dev/null | sed 's/,//g' | awk '{print $3}')
    if [ -z "$DOCKER_VERSION" ]; then
        DOCKER_VERSION="N/A"
        DOCKER_WARNING="docker is not installed, please install it!"
        WARNING_LOG+=("$DOCKER_WARNING")
    fi

    # 11. Check MTLink status
    mtlink_status=$(mthreads-gmi mtlink -s)
    if [ $? -eq 0 ]; then
        MTLINK_STATUS="enabled"
    fi

    # 12. Print result
    echo "================================================"
    echo "==================MUSA  Report=================="
    echo "================================================"
    printf "%-32s %-20s\n" "CPU: " "$(lscpu | grep 'Model name' | awk -F ': ' '{print $2}' | tr -d ' ')"
    printf "%-32s %-20s\n" "Operation System: " "$SystemVersion"
    printf "%-32s %-20s\n" "Kernel Version: " "$KernelVersion"
    printf "%-32s %-20s\n" "IOMMU: " "$(printf "$IOMMU_STATUS")"
    printf "%-32s %-20s\n" "lspci GPU Number: " "$GPUDevices"
    printf "%-32s %-20s\n" "PCIE version: " "$PCIE_VERSION"
    printf "MTBios Version: \n"
    IFS=$'\n' read -rd '' -a version_array <<<"$MTBios_VERSION"
    for version in "${version_array[@]}"; do
        printf "%-32s %-20s\n" " " "$version"
    done
    printf "%-32s %-20s\n" "MTLink Status: " "$MTLINK_STATUS"
    printf "%-32s %-20s\n" "Driver Version(gmi): " "$DRIVER_VERSION_GMI"
    printf "%-32s %-20s\n" "Detail Driver Version(clinfo): " "$DRIVER_VERSION_CLINFO"
    printf "%-32s %-20s\n" "mthreads-gmi Version" "$GMI_VERSION"
    printf "%-32s %-20s\n" "Docker Version: " "$DOCKER_VERSION"
    printf "%-32s %-20s\n" "MT-Container-Toolkit Version: " "$mt_container_toolkit_version"
    printf "%-32s %-20s\n" "MT-Container-Toolkit Status: " "$(printf "$MT_CONTAINER_TOOLKIT_STATUS")"
    echo "================================================"
    array_length=${#WARNING_LOG[@]}
    count=1
    if [ $array_length -gt 0 ]; then
        echo -e "${RED}Warning:${NC}"
        for i in "${WARNING_LOG[@]}"; do
            echo -e "${RED}$count.${NC} $i"
            ((count++))  
        done
    fi
}

function check_in_container() {
    if [ -f "/.dockerenv" ]; then
        echo -e "${RED}Warning: This script is Running inside a container. Running on host machine would be better!${NC}"
    fi
}

check_in_container

check_host_env
