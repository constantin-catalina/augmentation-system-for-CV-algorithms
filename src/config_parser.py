from dataclasses import dataclass, field
import augmentations


class ConfigError(Exception):
    """Raised when the config file is invalid"""


@dataclass
class Operation:
    name: str
    func: object
    args: list = field(default_factory=list)


def parse_line(line, line_no=0):

    chain = []
    for part in line.split(","):
        tokens = part.split()
        if not tokens:
            raise ConfigError(f"line {line_no}: empty operation in chain: {line!r}")
        aug = augmentations.lookup(tokens[0])
        if aug is None:
            known = ", ".join(a.name for a in augmentations.REGISTRY.values())
            raise ConfigError(
                f"line {line_no}: unknown augmentation {tokens[0]!r} (known: {known})"
            )
        try:
            args = augmentations.convert_params(aug, tokens[1:])
        except augmentations.ParamError as e:
            raise ConfigError(f"line {line_no}: {e}")
        chain.append(Operation(aug.name, aug.func, args))
    return chain


def load_config(path):

    steps = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            steps.append(parse_line(line, line_no))
    if not steps:
        raise ConfigError(f"no augmentation steps found in {path}")
    return steps
