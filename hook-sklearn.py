from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('sklearn') + collect_submodules('scipy')
