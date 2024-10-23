#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

PCIE_VERSION=''


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

pcie_version

echo "============================="
echo "PCIE version: ${PCIE_VERSION}"
echo "-----------------------------"
echo "Detailed information:"
sudo lspci -vvvvs ${first_pci_address} |grep Lnk
echo "============================="
