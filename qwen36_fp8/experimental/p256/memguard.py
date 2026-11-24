"""Run a command and kill its process tree if memory gets too high.
usage: memguard.py <tree_rss_gib> <cgroup_anon_gib> <log> -- cmd...
Logs peak tree RSS every minute. The container cgroup is read-only, so this replaces a memory.max."""
import os, signal, subprocess, sys, time

tree_lim, anon_lim, log = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
cmd = sys.argv[sys.argv.index('--') + 1:]
G = 2 ** 30

def children(pid):
    out = {pid}
    for d in os.listdir('/proc'):
        if not d.isdigit(): continue
        try:
            ppid = int(open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()[1])
        except OSError:
            continue
        if ppid == pid: out |= children(int(d))
    return out

def rss(pids):
    t = 0
    for p in pids:
        try:
            for line in open(f'/proc/{p}/status'):
                if line.startswith('VmRSS:'): t += int(line.split()[1]) * 1024
        except OSError:
            pass
    return t

def anon():
    for line in open('/sys/fs/cgroup/memory.stat'):
        if line.startswith('anon '): return int(line.split()[1])
    return 0

lf = open(log, 'a')
p = subprocess.Popen(cmd, start_new_session=True)
start = last = time.time(); peak = 0; killed = None
while p.poll() is None:
    r, a = rss(children(p.pid)), anon()
    peak = max(peak, r)
    if a > anon_lim * G or (tree_lim and r > tree_lim * G):
        killed = f'tree {r / G:.1f} GiB, cgroup anon {a / G:.1f} GiB'
        os.killpg(p.pid, signal.SIGKILL)
        break
    if time.time() - last > 60:
        last = time.time()
        print(f'{(last - start) / 60:.0f} min tree {r / G:.1f} GiB peak {peak / G:.1f} GiB anon {a / G:.1f} GiB', file=lf, flush=True)
    time.sleep(2)
rc = p.wait()
print(f'END rc {rc} after {(time.time() - start) / 60:.1f} min, peak {peak / G:.1f} GiB' + (f', KILLED at {killed}' if killed else ''), file=lf, flush=True)
sys.exit(rc)
