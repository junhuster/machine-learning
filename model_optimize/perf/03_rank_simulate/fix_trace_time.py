#!/usr/bin/env python3
"""修复 PyTorch/Kineto 导出的 chrome trace 在查看器里"时间轴按年显示"的问题。

原理：查看器用 ts 的最小值~最大值作为横轴范围。只要 trace 里混入一个量级异常的事件
（ts=0 的元数据事件、离群的 ts、CUPTI 异常导致的负 dur 或超大 dur），
整条时间轴就会被拉到几十年，真实的几个 step 被压成右边一条看不见的缝。

两种用法
--------
命令行：
  python3 fix_trace_time.py trace_chrome.json                 # 诊断 + 剔除离群 + 平移到 0
  python3 fix_trace_time.py trace_chrome.json --ns-to-us      # 时间戳单位是纳秒时额外换算
  python3 fix_trace_time.py trace_chrome.json --keep-outliers # 只平移，不删任何事件
  （输出 <同名>_fixed.json；加 --in-place 则原地覆盖）

被脚本 import（推荐，profile 完直接修，不用记第二件事）：
  from fix_trace_time import normalize_chrome_trace
  normalize_chrome_trace('trace_chrome.json')          # 自动判定单位、剔除坏事件

只读分析，不依赖 torch。
"""
import json
import os
import statistics as st
import sys

# 判定阈值（单位：trace 原始时间戳单位，通常是微秒）
TS_OUTLIER = 1e9     # 与中位数相差超过 1e9 视为离群（约 1000 秒）
DUR_MAX = 1e12       # 超过 1e12 视为异常时长（约 11 天）
SPAN_NS_HINT = 1e12  # span 超过此值基本可判定原始单位是纳秒


def normalize_chrome_trace(
    path,
    *,
    drop_outliers=True,
    ns_to_us=None,
    inplace=False,
    out_path=None,
    verbose=True,
):
    """就地/另存修复 trace，返回诊断统计 dict。"""
    with open(path) as f:
        data = json.load(f)
    events = data.get('traceEvents', [])
    if not events:
        raise ValueError('traceEvents 为空，检查文件是否完整')

    ts_all = [e['ts'] for e in events if isinstance(e.get('ts'), (int, float))]
    if not ts_all:
        raise ValueError('没有任何带 ts 的事件')
    durs = [e['dur'] for e in events
            if isinstance(e.get('dur'), (int, float)) and e['dur'] >= 0]

    span_before = max(ts_all) - min(ts_all)
    med_ts = st.median(ts_all)

    # --- 单位判定 ---
    auto_ns = span_before > SPAN_NS_HINT and (not durs or st.median(durs) >= 1e4)
    if ns_to_us is None:
        use_ns2us = auto_ns
        how = '自动判定' if auto_ns else '无需换算'
    else:
        use_ns2us = bool(ns_to_us)
        how = '显式指定'
    scale = 1000.0 if use_ns2us else 1.0

    # --- 找出把时间轴拉爆的事件 ---
    bad_idx, suspects = set(), []
    for i, e in enumerate(events):
        t, du = e.get('ts'), e.get('dur')
        if isinstance(t, (int, float)) and abs(t - med_ts) > TS_OUTLIER:
            bad_idx.add(i)
            suspects.append(('ts', t, e))
        elif isinstance(du, (int, float)) and (du < 0 or du > DUR_MAX):
            bad_idx.add(i)
            suspects.append(('dur', du, e))

    # --- 重写 ---
    keep, dropped = [], 0
    for i, e in enumerate(events):
        if drop_outliers and i in bad_idx:
            dropped += 1
            continue
        keep.append(e)

    base = min(e['ts'] for e in keep if isinstance(e.get('ts'), (int, float)))
    for e in keep:
        if isinstance(e.get('ts'), (int, float)):
            e['ts'] = (e['ts'] - base) / scale
        if isinstance(e.get('dur'), (int, float)):
            e['dur'] = e['dur'] / scale

    if scale != 1.0:
        data['displayTimeUnit'] = 'ms'
    data['traceEvents'] = keep

    if inplace:
        dst = path
    else:
        dst = out_path or (os.path.splitext(path)[0] + '_fixed.json')
    with open(dst, 'w') as f:
        json.dump(data, f)

    span_after = max(e['ts'] for e in keep if isinstance(e.get('ts'), (int, float)))
    unit = 'ms' if scale != 1.0 else 'us'
    stats = {
        'events': len(events),
        'dropped': dropped,
        'scale': scale,
        'span_before': span_before,
        'span_after': span_after,
        'unit': unit,
        'output': dst,
        'suspects': suspects,
    }

    if verbose:
        print(f'events={len(events)}  displayTimeUnit={data.get("displayTimeUnit")}')
        print(f'ts  min={min(ts_all):.4g}  med={med_ts:.4g}  max={max(ts_all):.4g}')
        print(f'span_before={span_before:.4g}  '
              f'(若单位是微秒则为 {span_before / 1e6:.4g} 秒)')
        if durs:
            print(f'dur med={st.median(durs):.4g}  max={max(durs):.4g}')
        if scale != 1.0:
            print(f'>> 判定时间戳单位为纳秒（{how}），已除以 1000 转微秒')
        for kind, v, e in suspects[:10]:
            print(f'  SUSPECT {kind}={v:.4g}  cat={e.get("cat")}  '
                  f'name={str(e.get("name"))[:70]}')
        if len(suspects) > 10:
            print(f'  ... 共 {len(suspects)} 个可疑事件')
        print(f'-> {dst}  dropped={dropped}  scale=1/{scale:g}  '
              f'span_after={span_after:.4g} {unit}')
    return stats


def _parse_argv(argv):
    flags = {a for a in argv if a.startswith('--')}
    args = [a for a in argv if not a.startswith('--')]
    return (args[0] if args else 'trace_chrome.json'), flags


def main():
    src, flags = _parse_argv(sys.argv[1:])
    normalize_chrome_trace(
        src,
        drop_outliers='--keep-outliers' not in flags,
        ns_to_us=True if '--ns-to-us' in flags else None,
        inplace='--in-place' in flags,
    )
    print('用输出文件重新打开查看器；若仍不对，把上面的诊断输出发出来。')


if __name__ == '__main__':
    main()
