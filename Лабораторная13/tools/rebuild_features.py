"""Восстановить общий NPZ из отдельных NPY без повторного извлечения Inception."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def rebuild(folder):
    name='feat13_disc_color.npz'
    target=folder/name
    expected=json.loads((folder/'artifact_hashes.json').read_text())[name]
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError('Существующий NPZ отличается от исходного; автоматическая перезапись отключена')
        print('Уже восстановлен, SHA-256 совпадает:',target)
        return
    keys=list(json.loads((folder/'feature_shapes.json').read_text()))
    parts=[(key,folder/f'part13_disc_color_{key}.npy') for key in keys]
    for _,part in parts:
        if not part.is_file():raise ValueError('Не найден '+str(part))
    temp=target.with_suffix('.npz.tmp')
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as archive:
            for key,part in parts:
                info=zipfile.ZipInfo(key+'.npy',date_time=(1980,1,1,0,0,0))
                with part.open('rb') as src,archive.open(info,'w',force_zip64=True) as dest:
                    shutil.copyfileobj(src,dest)
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=expected:
            raise ValueError('Восстановленный архив не совпал с исходной контрольной суммой')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    print('Восстановлен, SHA-256 совпадает:',target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    try:rebuild(args.folder)
    except ValueError as error:parser.exit(1,str(error)+'\n')
