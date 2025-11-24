# tools/parser.py
import re
import yaml
import json
from typing import Dict, Optional, Any, Tuple

FRONT_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
NODE_ATTR_RE = re.compile(r"(?P<id>\w+)@\{(?P<attrs>[^}]*)\}")
EDGE_RE = re.compile(r"(?P<src>\w+)\s*([-=.]*>+)\s*(?:\|(?P<label>[^|]+)\|\s*)?(?P<dst>\w+)" )
KV_RE = re.compile(r"(?P<k>[\w\-]+)\s*:\s*(?P<v>[^,]+)(?:,|$)")
NODE_TYPE_RE = re.compile(r"(?P<type>\w+)\s*(?:\((?P<params>.*)\))?")


def parse_frontmatter(text: str) -> Tuple[Optional[dict], str]:
    m = FRONT_RE.match(text)
    if not m:
        return None, text
    fm_raw = m.group(1)
    rest = text[m.end():]
    try:
        fm = yaml.safe_load(fm_raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse YAML frontmatter: {e}")
    return fm, rest


def parse_node_attrs(attrs_raw: str) -> Dict[str, Any]:
    # parse simple comma-separated key: value pairs
    out = {}
    for m in KV_RE.finditer(attrs_raw):
        k = m.group('k').strip()
        v = m.group('v').strip()
        # strip quotes
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        # node: TypeName(args)
        if k == 'node':
            mt = NODE_TYPE_RE.match(v)
            if mt:
                t = mt.group('type')
                p = mt.group('params')
                params = {}
                if p:
                    # simple params parsing: key=val, flag
                    for part in re.split(r'\s*,\s*', p):
                        if '=' in part:
                            kk, vv = part.split('=', 1)
                            params[kk.strip()] = yaml.safe_load(vv.strip())
                        else:
                            params[part.strip()] = True
                out['type'] = t
                out['params'] = params
                continue
        # params: key=val, flag - parse as parameter dictionary
        if k == 'params':
            params = {}
            # simple params parsing: key=val, flag
            for part in re.split(r'\s*,\s*', v):
                if '=' in part:
                    kk, vv = part.split('=', 1)
                    params[kk.strip()] = yaml.safe_load(vv.strip())
                else:
                    params[part.strip()] = True
            out[k] = params
            continue
        out[k] = yaml.safe_load(v)
    return out


def parse_mermaid_flow(body: str) -> Dict[str, Any]:
    nodes = {}
    edges = []
    # Node attrs first
    for m in NODE_ATTR_RE.finditer(body):
        nid = m.group('id')
        attrs = m.group('attrs')
        parsed = parse_node_attrs(attrs)
        nodes[nid] = {
            'id': nid,
            'shape_hint': parsed.pop('shape', 'rect'),
            'type': parsed.pop('type'),
            'params': parsed.pop('params', {}),
            'label': parsed.pop('label', nid),
            'meta': parsed,
        }
    # Edges
    for m in EDGE_RE.finditer(body):
        edges.append({
            'source': m.group('src'),
            'target': m.group('dst'),
            'kind': m.group(2),
            'label': m.group('label') and m.group('label').strip(),
        })
    return {'nodes': list(nodes.values()), 'edges': edges}


def parse_file(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        raw = f.read()
    front, body = parse_frontmatter(raw)
    ir = {
        'id': path,
        'meta': {
            'frontmatter': front
        },
    }
    mer = parse_mermaid_flow(body)
    ir.update(mer)
    # harvest variables referenced in the body
    vars_found = set(re.findall(r"\$\{?([A-Za-z0-9_\.\-/]+)\}?", body))
    ir['variables'] = sorted(vars_found)
    return ir


if __name__ == '__main__':
    import sys
    p = sys.argv[1]
    ir = parse_file(p)
    print(json.dumps(ir, indent=2))
