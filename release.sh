#!/bin/bash

pip install pyinstaller

pyinstaller --noconfirm --onefile --console --name "dooffice" --add-data "module:module" --add-data "README.md:." main.py

echo "Success! Executable file located at dist/dooffice.exe"