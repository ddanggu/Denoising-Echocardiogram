from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, Iterable, List


def mk_dir(paths: Iterable[str | Path]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def _short(v: Any, max_len: int = 60) -> str:
    s = str(v)
    if len(s) <= max_len:
        return s
    return '...' + s[-(max_len - 3):]

def _title_from_class(cls_name: str) -> str:
    # CamelCase -> "Camel Case" (simple)
    out = []
    for i, ch in enumerate(cls_name):
        if i > 0 and ch.isupper() and (not cls_name[i - 1].isupper()):
            out.append(" ")
        out.append(ch)
    return "".join(out)


@dataclass
class BaseSettings:
    name: Optional[str] = None
    data: Optional[str] = None
    
    mkdir:      bool = False
    dtype:      str = 'float64'
    device:     str = 'cpu'
    multi_gpu:  bool = False

@dataclass
class Paths:
    base_path:  str | Path = Path("/ds/mkseo/SM-Dehazing")
    
    data_path:  str | Path = field(init=False)
    res_path:   str | Path = field(init=False)
    fig_path:   str | Path = field(init=False)
    loss_path:  str | Path = field(init=False)
    model_path: str | Path = field(init=False)

@dataclass
class TrainConfig:
    model: str = "gan"
    clean_dir: str = ""
    haze_dir: str = ""
    save_dir: str = "res/train"

    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    image_size: int | None = None
    value_range: str = "0_1"

    in_ch: int = 1
    out_ch: int = 1
    depth: int = 4
    g_init_ch: int = 64
    d_init_ch: int = 64
    out_act: str | None = "sigmoid"
    norm: str | None = "batch"

    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    gan_mode: str = "bce"

    lambda_gan: float = 1.0
    lambda_pair: float = 100.0
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.0

    amp: bool = False
    grad_clip: float | None = 3.0
    save_every: int = 10
    device: str = "cuda"

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))

# ----------------------------
# Main Config (hierarchical)
# ----------------------------
@dataclass(init=False)
class Config:
    # sections
    base:  BaseSettings = field(default_factory=BaseSettings)
    paths: Paths        = field(default_factory=Paths)
    train: TrainConfig  = field(default_factory=TrainConfig)

    __section_order__ = ("base", "paths", "train")
    __shared_fields__ = ("device",)

    def __init__(self, **kwargs):
        # 1) 섹션 기본 생성
        object.__setattr__(self, "base", BaseSettings())
        object.__setattr__(self, "paths", Paths())
        object.__setattr__(self, "train", TrainConfig())

        # 2) flat kwargs를 섹션으로 자동 분배
        #    - section 이름으로 dict/dataclass를 넘기면 해당 section에 직접 반영
        for key, value in kwargs.items():
            if key in {"base", "paths", "train"}:
                self._update_section(key, value)
                continue

            hits = []
            for sec_name, sec in self._iter_sections():
                if hasattr(sec, key):
                    hits.append((sec_name, sec))

            if len(hits) == 0:
                raise TypeError(f"Config.__init__() got an unexpected keyword argument '{key}'")
            if len(hits) > 1:
                if key in self.__shared_fields__:
                    self._set_shared_field(key, value)
                    continue
                raise TypeError(
                    f"Ambiguous config key '{key}': appears in multiple sections. "
                    f"Use explicit section assignment (e.g., cfg.train.{key}=...)"
                )

            setattr(hits[0][1], key, value)

        # 3) derived 계산
        self.__post_init__()

    def __post_init__(self):
        self._sync_shared_fields()

        name = self.base.name or "Temp"
        base = Path(self.paths.base_path)
        res_path = base / "res" / name

        self.paths.data_path    = base / "data"
        self.paths.res_path     = res_path
        self.paths.model_path   = res_path / "model"
        self.paths.fig_path     = res_path / "fig"
        self.paths.loss_path    = res_path / "loss"

        if self.base.mkdir:
            mk_dir([self.paths.res_path, self.paths.model_path, self.paths.fig_path, self.paths.loss_path])


    def _update_section(self, section_name: str, values: Any):
        section = getattr(self, section_name)

        if is_dataclass(values) and not isinstance(values, type):
            values = asdict(values)
        if not isinstance(values, dict):
            raise TypeError(f"Config section '{section_name}' must be a dict or dataclass")

        for key, value in values.items():
            if not hasattr(section, key):
                raise TypeError(f"Config section '{section_name}' got an unexpected key '{key}'")
            setattr(section, key, value)
            if key in self.__shared_fields__:
                self._set_shared_field(key, value)

    def _set_shared_field(self, name: str, value: Any):
        for _, section in self._iter_sections():
            if hasattr(section, name):
                setattr(section, name, value)

    def _sync_shared_fields(self):
        for name in self.__shared_fields__:
            for _, section in self._iter_sections():
                if hasattr(section, name):
                    self._set_shared_field(name, getattr(section, name))
                    break

    def _iter_sections(self):
        name_to_obj = {}
        for f in fields(type(self)):
            try:
                v = object.__getattribute__(self, f.name)
            except AttributeError:
                continue
            if is_dataclass(v):
                name_to_obj[f.name] = v

        order = getattr(type(self), "__section_order__", None)
        if order:
            for name in order:
                if name in name_to_obj:
                    yield name, name_to_obj.pop(name)

        for name, obj in name_to_obj.items():
            yield name, obj

    def __getattr__(self, name):
        for _, section in self._iter_sections():
            if hasattr(section, name):
                return getattr(section, name)
        raise AttributeError(f"'Config' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        cfg_field_names = {f.name for f in fields(self)} if is_dataclass(self) else set()
        if name in cfg_field_names:
            object.__setattr__(self, name, value)
            return

        if name in self.__shared_fields__:
            self._set_shared_field(name, value)
            return

        for f in fields(self):
            if f.name in self.__dict__:
                section = self.__dict__[f.name]
                if is_dataclass(section) and hasattr(section, name):
                    setattr(section, name, value)
                    
                    if f.name in ("base", "paths") and name in ("name", "base_path"):
                        self.__post_init__()
                    return

        object.__setattr__(self, name, value)

    def to_dict(self, include_derived: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if include_derived:
            d.update({
                "res_path": self.res_path,
                "model_path": self.model_path,
                "fig_path": self.fig_path,
                "loss_path": self.loss_path,
            })
        return d
    
    def __str__(self) -> str:
        section_fields = []
        for f in fields(self):
            v = getattr(self, f.name, None)
            if is_dataclass(v):
                section_fields.append(f)

        lines: List[str] = []
        lines.append("-" * 70)
        lines.append(f"{'Config Details (Auto)':^70}")
        lines.append("-" * 70)

        for f in section_fields:
            section_obj = getattr(self, f.name)
            title = f.metadata.get("section") if f.metadata else None
            if not title:
                title = _title_from_class(type(section_obj).__name__)

            lines.append("")
            lines.append(f"[ {title} ]")

            for sf in fields(section_obj):
                if getattr(sf, "repr", True) is False:
                    continue
                val = getattr(section_obj, sf.name)
                lines.append(f"  {sf.name:<22} : {_short(val)}")

        lines.append("-" * 70)
        return "\n".join(lines)
