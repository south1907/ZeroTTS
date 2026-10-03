"""Precomputed voice packs.

A "voice" in ZeroTTS is a small array of speaker latents — shape
``(1, n_voice_queries, d_model)`` float32 — that gets prepended to the sequence.
That array is the *entire* speaker conditioning; there is no reference audio, no
transcript, and no prompt frames involved at generation time.

Those latents are produced by a voice encoder that reads a reference clip. **The
encoder is not part of this release**, so this package cannot create a voice
from a wav — it can only load ones that already exist. See the README, or
zeroweight.ai, for how to obtain latents for your own speaker.

That boundary is narrower than it sounds: a voice pack is just a .npz, so
latents obtained elsewhere drop into ``voices/<name>/voice.npz`` and work with
no code change. A pack does not even have to live under the model directory:
:func:`install_voice_zip` takes the zip a zeroweight.ai voice downloads as and
unpacks it into :data:`USER_VOICES_DIR`, where every later ``ZeroTTS`` picks it
up on its own — which is what :meth:`ZeroTTS.add_voices` calls.

Layout of a voice pack directory:

    voices/
      index.json                 # optional manifest: name, display_name, tags...
      <name>/
        voice.npz                # required: {n_voice_queries: int64, voice_emb: (1,Q,D) f32}
        voice.bin                # optional: raw f32 of voice_emb, for the JS demo
        preview.wav              # optional
        meta.json                # optional: display_name, gender, tags, description...
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Where installed voices live — a downloaded pack is copied here, not merely
#: pointed at, so it survives the tab/process that installed it and is found by
#: every later ``ZeroTTS``. Deliberately NOT inside the model directory: that is
#: a cache of a read-only repo, and a re-download would take the voice with it.
USER_VOICES_DIR = Path(
    os.environ.get("ZEROTTS_VOICES_HOME") or "~/.zerotts/voices").expanduser()


@dataclass
class Voice:
    """A loaded voice pack."""

    name: str
    emb: np.ndarray            # (1, n_voice_queries, d_model) float32
    meta: dict
    preview_path: str | None = None

    @property
    def n_voice_queries(self) -> int:
        return int(self.emb.shape[1])

    @property
    def language(self) -> str:
        return str(self.meta.get("language", "vi"))

    @property
    def description(self) -> str:
        return str(self.meta.get("description", ""))

    @property
    def display_name(self) -> str:
        """Human-readable name, e.g. "Mai Chi" for the pack directory "maichi"."""
        return str(self.meta.get("display_name", self.name))

    @property
    def gender(self) -> str:
        return str(self.meta.get("gender", ""))

    @property
    def tags(self) -> list[str]:
        """Free-form labels — gender, age, register, tone ("nữ", "trẻ", "kể
        chuyện", "ấm áp"...) — for filtering or displaying a voice picker."""
        return list(self.meta.get("tags", []))


def is_voice_pack(path: str | Path) -> bool:
    """Whether ``path`` is itself a voice pack directory, i.e. holds a
    ``voice.npz``. The one file that makes a directory a voice."""
    return (Path(path) / "voice.npz").exists()


def _voice_dirs(voices_root: Path):
    if not voices_root.is_dir():
        return []
    return sorted(d for d in voices_root.iterdir() if d.is_dir() and is_voice_pack(d))


def discover_packs(path: str | Path) -> dict:
    """``{name: directory}`` for every voice pack at or under ``path``.

    Both shapes people actually have are accepted, because both are what a
    download hands them: the pack directory itself (``.../maichi/`` with a
    ``voice.npz`` in it), or a directory holding several of those. Nothing is
    searched recursively beyond that — a deep scan of an arbitrary directory is
    a surprising thing for a path argument to do.
    """
    path = Path(path).expanduser()
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")
    if is_voice_pack(path):
        return {path.name: path}
    return {d.name: d for d in _voice_dirs(path)}


#: The files a pack is made of. Anything else in the zip is ignored — a
#: download carries a README, and a folder that has been opened on a Mac
#: carries __MACOSX/ and .DS_Store noise that must not land in the install.
PACK_FILES = ("voice.npz", "voice.bin", "preview.wav", "meta.json")


def _zip_packs(archive: zipfile.ZipFile, fallback_name: str) -> dict:
    """``{name: {filename: member}}`` for every pack in an open zip.

    A pack is a directory holding a ``voice.npz``. The download ships exactly
    one, under its own name, but a zip of several is the obvious thing for
    someone to build and costs nothing to accept. A zip whose ROOT is the pack
    is named after the file, since there is no directory to take a name from.
    """
    packs: dict = {}
    for member in archive.infolist():
        if member.is_dir():
            continue
        parts = Path(member.filename.replace("\\", "/")).parts
        # Mac Finder writes a parallel __MACOSX/ tree of resource forks whose
        # paths mirror the real ones — following them would "find" every pack
        # twice, the second time with unreadable files.
        if not parts or parts[0] == "__MACOSX":
            continue
        base = parts[-1]
        if base not in PACK_FILES:
            continue
        name = parts[-2] if len(parts) > 1 else fallback_name
        packs.setdefault(name, {})[base] = member
    return {name: files for name, files in packs.items() if "voice.npz" in files}


def install_voice_zip(zip_path: str | Path, dest_root: str | Path | None = None) -> dict:
    """Unpack a downloaded voice zip into ``dest_root``. Returns ``{name: dir}``.

    This is the whole "I have the file, make it a voice" step: the archive is
    read, each pack inside it is written to ``dest_root/<name>/``, and the
    result is a directory :func:`load_voice_dir` accepts. Members are extracted
    by name into a directory we choose rather than with ``extractall``, so a
    crafted archive cannot write outside it.

    An existing voice of the same name is REPLACED. Re-downloading a voice after
    re-cloning it is the common case, and two voices differing only by a ``(1)``
    suffix would be worse than one that is simply current.
    """
    zip_path = Path(zip_path).expanduser()
    dest_root = Path(dest_root).expanduser() if dest_root else USER_VOICES_DIR
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path.name} is not a zip file")

    with zipfile.ZipFile(zip_path) as archive:
        packs = _zip_packs(archive, fallback_name=zip_path.stem)
        if not packs:
            raise FileNotFoundError(
                f"{zip_path.name} holds no voice pack — expected a folder with a "
                "voice.npz in it")

        installed = {}
        for name, files in packs.items():
            # The name becomes a directory name, so it has to survive being one.
            safe = "".join(c for c in name if c.isalnum() or c in "._- ").strip() or "voice"
            vdir = dest_root / safe
            if vdir.exists():
                shutil.rmtree(vdir)
            vdir.mkdir(parents=True)
            for base, member in files.items():
                with archive.open(member) as src, open(vdir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            installed[safe] = vdir
    return installed


def installed_voices(dest_root: str | Path | None = None) -> dict:
    """``{name: directory}`` for every voice installed with
    :func:`install_voice_zip`. Empty before the first one."""
    root = Path(dest_root).expanduser() if dest_root else USER_VOICES_DIR
    return {d.name: d for d in _voice_dirs(root)}


def list_voices(voices_root: str | Path) -> list:
    """Voice names available under ``voices_root``, sorted."""
    return [d.name for d in _voice_dirs(Path(voices_root))]


def load_voice(voices_root: str | Path, name: str, expect_queries: int | None = None) -> Voice:
    """Load one voice pack by name from ``voices_root``.

    ``expect_queries`` is the model's ``n_voice_queries``. A mismatch is fatal
    and says so: latents built for a different model would still be the right
    dtype and rank, so they would feed the graph cleanly and produce confident
    nonsense.
    """
    root = Path(voices_root)
    vdir = root / name
    if not is_voice_pack(vdir):
        available = list_voices(root)
        raise FileNotFoundError(
            f"no voice {name!r} in {root} (available: {available or 'none'})")
    return load_voice_dir(vdir, name, expect_queries=expect_queries)


def load_voice_dir(vdir: str | Path, name: str | None = None,
                   expect_queries: int | None = None) -> Voice:
    """Load the voice pack in directory ``vdir``.

    The directory-addressed half of :func:`load_voice`, so a pack that lives
    anywhere — a folder someone downloaded, not a subdirectory of the model —
    loads by the same code path.
    """
    vdir = Path(vdir)
    name = name or vdir.name
    npz = vdir / "voice.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{vdir} is not a voice pack: no voice.npz in it")

    data = np.load(npz)
    emb = np.asarray(data["voice_emb"], dtype=np.float32)
    if emb.ndim == 2:
        emb = emb[None, :, :]

    stored_q = int(data["n_voice_queries"]) if "n_voice_queries" in data else int(emb.shape[1])
    if stored_q != emb.shape[1]:
        raise ValueError(
            f"voice {name!r} is inconsistent: n_voice_queries={stored_q} but "
            f"voice_emb has {emb.shape[1]} queries.")
    if expect_queries is not None and stored_q != expect_queries:
        raise ValueError(
            f"voice {name!r} was built with n_voice_queries={stored_q}, but this "
            f"model uses {expect_queries}. It belongs to different weights.")

    meta_path = vdir / "meta.json"
    # UTF-8 required: Windows default locale encoding garbles Vietnamese meta.
    meta = (json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists() else {})
    preview = vdir / "preview.wav"
    return Voice(name=name, emb=emb, meta=meta,
                 preview_path=str(preview) if preview.exists() else None)


def load_index(voices_root: str | Path) -> dict:
    """The ``index.json`` manifest, or a minimal one synthesized from the dirs."""
    root = Path(voices_root)
    index_path = root / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return {"voices": [{"name": n} for n in list_voices(root)]}
