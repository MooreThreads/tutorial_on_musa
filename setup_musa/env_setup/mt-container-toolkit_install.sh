#!/bin/bash


# Conditional on mt-container-toolkit being a zip compressed package
mt_container_toolkit_install(){
    if [ -z $1 ]; then
        echo "Usage: $0 <mt-container-toolkit dir path>, eg: bash ./mt-container-toolkit_install.sh ./mt-container-toolkit-1.9.0/"
    else

        # 1. install 
        for deb in "mtml" "sgpu-dkms" "mt-container-toolkit"
        do
            package_file=$(find $1 -maxdepth 1 -type f -name "*${deb}*")
            echo "# installing $package_file"
            sudo dpkg -i $package_file
            if [ ! $? -eq 0 ]; then
                # Red font
                echo $package_file
                echo -e "\033[31mWarning: \033[0munable to install $package_file, please check your local environment"
                exit 1
            fi
        done

        # 2. bind
        (cd /usr/bin/musa && sudo ./docker setup $PWD)
        
        # 3. check
        docker run --rm --env MTHREADS_VISIBLE_DEVICES=all registry.mthreads.com/cloud-mirror/ubuntu:20.04 mthreads-gmi 2> /dev/null
        if [ ! $? -eq 0 ]; then
            # Red font
            echo -e "\033[31mWarning: \033[0mdocker run --rm --env MTHREADS_VISIBLE_DEVICES=all registry.mthreads.com/cloud-mirror/ubuntu:20.04 mthreads-gmi failed. There may be a problem binding the Moore thread container runtime to Docker, please refer to https://docs.mthreads.com/cloud-native/cloud-native-doc-online/install_guide"
            exit 1
        else    
            echo -e "\033[32mmt-container-toolkit is installed successfully.\033[0m"
        fi
    fi
}

mt_container_toolkit_install $1
