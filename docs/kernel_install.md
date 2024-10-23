1. 查看当前版本

   ```shell
   $ uname -r
   4.15.0-101-generic
   
   $ lsb_release -a
   No LSB modules are available.
   Distributor ID: Ubuntu
   Description:    Ubuntu 18.04.4 LTS
   Release:        18.04
   Codename:       bionic
   ```

2. 查看当前已安装的Kernel Image.

   ```shell
   $ dpkg --get-selections |grep linux-image
   linux-image-4.15.0-101-generic                  install
   linux-image-generic                             install
   ```

3. 查询当前软件仓库可以安装的 Kernel Image 版本，如果没有预期的版本，则需要额外配置仓库.

   ```shell
   $ apt-cache search linux | grep linux-image
   ```

4. 安装指定版本的 Kernel Image 和 Kernel Header.

   ```shell
   $ sudo apt-get install linux-headers-4.15.0-76-generic linux-image-4.15.0-76-generic
   ```

5. 查看当前的 Kernel 列表.

   ```shell
   $ grep menuentry /boot/grub/grub.cfg
   if [ x"${feature_menuentry_id}" = xy ]; then
     menuentry_id_option="--id"
     menuentry_id_option=""
   export menuentry_id_option
   menuentry 'Ubuntu' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-b753ddfd-2731-4c32-aa02-9a654abc99c6' {
   submenu 'Advanced options for Ubuntu' $menuentry_id_option 'gnulinux-advanced-b753ddfd-2731-4c32-aa02-9a654abc99c6' {
           menuentry 'Ubuntu, with Linux 4.15.0-101-generic' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.15.0-101-generic-advanced-b753ddfd-2731-4c32-aa02-9a654abc99c6' {
           menuentry 'Ubuntu, with Linux 4.15.0-101-generic (recovery mode)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.15.0-101-generic-recovery-b753ddfd-2731-4c32-aa02-9a654abc99c6' {
           menuentry 'Ubuntu, with Linux 4.15.0-76-generic' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.15.0-76-generic-advanced-b753ddfd-2731-4c32-aa02-9a654abc99c6' {
           menuentry 'Ubuntu, with Linux 4.15.0-76-generic (recovery mode)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-4.15.0-76-generic-recovery-b753ddfd-2731-4c32-aa02-9a654abc99c6' {
   ```

6. 修改 Kernel 的启动顺序：如果安装的是最新的版本，那么默认就是首选的；如果安装的是旧版本，就需要修改 grub 配置.

   ```shell
   $ vi /etc/default/grub
   
   # GRUB_DEFAULT=0
   GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 4.15.0-76-generic"
   ```

7. 生效配置

   ```shell
   $ update-grub
   $ reboot
   ```

