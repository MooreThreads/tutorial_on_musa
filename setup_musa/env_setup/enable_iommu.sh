#!/bin/bash

function iommu_enable() {
    # 1. get cpu architecture
    CPU_ARCHITECTURE=lscpu | grep "Architecture" | awk '{print $2}'
    # 2. get cpu mem
    CPU_MEMORY_TOTAL=`free -g | grep "Mem:" | awk '{print $2}'`
    # 3. get CPU Vendor ID
    CPU_VENDOR=`lscpu | grep "Vendor ID" | awk '{print $3}'`
    # 4. IOMMU status
    IOMMU_STATUS=`cat /var/log/dmesg | grep -e "AMD-Vi: Interrupt remapping enabled" -e "IOMMU enabled"`

    # 5. check
    if [[ "$IOMMU_STATUS"==*"enabled"* ]]; then
        echo "The IOMMU status of your current machine is enabled and no action is required."
        exit 0
    fi

    # 6. enable operation
    # 6.1 check architecture
    if [ $CPU_ARCHITECTURE="x86_64" ]; then
        # 6.2 check CPU memory
        if [ $CPU_MEMORY_TOTAL -ge 256 ]; then
            # 6.3 check CPU vendor id
            if echo "$CPU_VENDOR" | grep -iq "intel"; then
                sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on iommu.passthrough=0"/' /etc/default/grub
            elif "$CPU_VENDOR" | grep -iqE "amd|hg"; then
                sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="amd_iommu=on iommu.passthrough=0"/' /etc/default/grub
            else
                echo "Unable to enable IOMMU based on your CPU information, please enable it manually."
            fi
        else
            echo "It is detected that the CPU memory size of your machine is less than 256G. The IOMMU does not need to be enabled in this case."
        fi

    else
        echo "Your machine is not built on the x86_64 architecture, it is built on the  architecture, please ignore this script operation."
    fi
}

iommu_enable

