# Add the LLVM repository
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main"

# Update and install
sudo apt update
sudo apt install clang-22 lldb-22 lld-22

# Install libstdc++ development files
sudo apt update
sudo apt install libstdc++-12-dev

# # Or install the full build-essential package which includes everything
# sudo apt install build-essential

# # You might also need libc++ if you want to use LLVM's libc++
# sudo apt install libc++-dev libc++abi-dev

# # sudo ln -sf /usr/bin/ld.lld /usr/bin/ld.lld-22
# sudo ln -sf /usr/bin/ld.lld-22 /usr/bin/ld.lld

# sudo apt remove lld-22
# sudo apt purge lld-22
# sudo apt autoremove