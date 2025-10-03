# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('swing_model_enhanced.pkl', '.'), ('swing_scaler_enhanced.pkl', '.'), ('feature_columns_enhanced.pkl', '.')]
datas += collect_data_files('sklearn')


a = Analysis(
    ['swing_trading_system.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['sklearn.ensemble', 'sklearn.tree', 'sklearn.utils._weight_vector', 'sklearn.neighbors._partition_nodes', 'scipy.special._ufuncs_cxx', 'scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack'],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SwingTradingSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
