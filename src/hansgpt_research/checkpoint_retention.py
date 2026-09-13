"""Bound disk usage inside one newly created run directory."""

import re
from pathlib import Path


def prune_run_checkpoints(directory, *, keep_recent, retain_every, final_position):
    if keep_recent < 1 or retain_every < 1 or final_position < 1:
        raise ValueError('Retention requires positive intervals/counts')
    directory=Path(directory)
    if directory.is_symlink():
        raise ValueError('Refuse symlink checkpoint directory')
    candidates=[]
    for path in directory.iterdir():
        match=re.fullmatch(r'positions_(\d+)\.pt',path.name)
        if match and path.is_file():
            if path.is_symlink():
                raise ValueError('Refuse symlink checkpoint')
            candidates.append((int(match[1]),path))
    candidates.sort()
    keep={p for _,p in candidates[-keep_recent:]}
    keep.update(p for count,p in candidates if count%retain_every==0 or count==final_position)
    deleted=[]
    for _,path in candidates:
        if path not in keep:
            path.unlink()
            deleted.append(path.name)
    return deleted
