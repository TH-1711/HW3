import os
import csv
import time
import random

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. Mô hình bài toán  (CHÉP NGUYÊN từ notebook nhóm)
# ----------------------------------------------------------------------------
NUM_COURSES, NUM_SECTIONS, NUM_PROFESSORS = 15, 10, 10
NUM_DAYS, NUM_TIMESLOTS, NUM_ROOMS = 5, 6, 30
DAY_PAIRS = [(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)]


class Room:
    def __init__(self, room_id, size):
        self.id = room_id
        self.size = size


class ClassSection:
    def __init__(self, section_id, course_id, prof_id, size):
        self.id = section_id
        self.course_id = course_id
        self.prof_id = prof_id
        self.size = size


def build_problem(seed_setup=42):
    """Khởi tạo phòng & lớp - cố định seed để mọi cấu hình khảo sát cùng đề."""
    rnd = random.Random(seed_setup)
    rooms = [Room(i, rnd.choice([0, 1])) for i in range(NUM_ROOMS)]
    classes = []
    prof_course_count = {i: 0 for i in range(1, NUM_PROFESSORS + 1)}
    for section_id in range(25):
        course_id = rnd.randint(1, NUM_COURSES)
        available = [p for p, c in prof_course_count.items() if c < 3]
        if not available:
            break
        prof_id = rnd.choice(available)
        prof_course_count[prof_id] += 1
        classes.append(
            ClassSection(section_id, course_id, prof_id, rnd.choice([0, 1]))
        )
    return rooms, classes


class Schedule:
    def __init__(self, rooms, classes):
        self.rooms = rooms
        self.classes = classes
        self.genes = {}
        self.fitness = 0.0
        self.conflicts = 0

    def random_init(self, rnd):
        for c in self.classes:
            self.genes[c.id] = [
                rnd.randint(0, len(DAY_PAIRS) - 1),
                rnd.randint(0, NUM_TIMESLOTS - 1),
                rnd.randint(0, NUM_ROOMS - 1),
                rnd.randint(0, NUM_TIMESLOTS - 1),
                rnd.randint(0, NUM_ROOMS - 1),
            ]
        self.calculate_fitness()

    def calculate_fitness(self):
        self.conflicts = 0
        room_usage = set()
        prof_usage = set()
        for c in self.classes:
            dp_idx, slot1, room1_idx, slot2, room2_idx = self.genes[c.id]
            day1, day2 = DAY_PAIRS[dp_idx]
            if self.rooms[room1_idx].size < c.size:
                self.conflicts += 1
            if self.rooms[room2_idx].size < c.size:
                self.conflicts += 1
            for d, s, r in [(day1, slot1, room1_idx), (day2, slot2, room2_idx)]:
                if (d, s, r) in room_usage:
                    self.conflicts += 1
                else:
                    room_usage.add((d, s, r))
            for d, s in [(day1, slot1), (day2, slot2)]:
                if (d, s, c.prof_id) in prof_usage:
                    self.conflicts += 1
                else:
                    prof_usage.add((d, s, c.prof_id))
        self.fitness = 1.0 / (1.0 + self.conflicts)


def crossover(p1, p2, rnd):
    child = Schedule(p1.rooms, p1.classes)
    for c in p1.classes:
        child.genes[c.id] = (
            p1.genes[c.id].copy() if rnd.random() > 0.5 else p2.genes[c.id].copy()
        )
    return child


def mutate(sch, rnd, mutation_rate=0.1):
    limits = [len(DAY_PAIRS) - 1, NUM_TIMESLOTS - 1, NUM_ROOMS - 1,
              NUM_TIMESLOTS - 1, NUM_ROOMS - 1]
    for c in sch.classes:
        if rnd.random() < mutation_rate:
            idx = rnd.randint(0, 4)
            sch.genes[c.id][idx] = rnd.randint(0, limits[idx])
    sch.calculate_fitness()


# ----------------------------------------------------------------------------
# 2. Hàm chạy 1 lần GA
# ----------------------------------------------------------------------------
def run_ga(rooms, classes, seed,
           population_size=100, generations=500,
           mutation_rate=0.1, tournament_k=3):
    rnd = random.Random(seed)
    population = [Schedule(rooms, classes) for _ in range(population_size)]
    for p in population:
        p.random_init(rnd)

    best_fitness_history = []
    best_conflicts_history = []
    converged_at = None
    t0 = time.time()
    for gen in range(generations):
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness_history.append(population[0].fitness)
        best_conflicts_history.append(population[0].conflicts)

        if population[0].conflicts == 0 and converged_at is None:
            converged_at = gen
            break  # dừng sớm khi đạt 0 xung đột

        new_pop = [population[0]]                       # elitism
        while len(new_pop) < population_size:
            p1 = max(rnd.sample(population, tournament_k),
                     key=lambda x: x.fitness)
            p2 = max(rnd.sample(population, tournament_k),
                     key=lambda x: x.fitness)
            child = crossover(p1, p2, rnd)
            mutate(child, rnd, mutation_rate)
            child.calculate_fitness()
            new_pop.append(child)
        population = new_pop

    population.sort(key=lambda x: x.fitness, reverse=True)
    best = population[0]
    runtime = time.time() - t0

    return {
        "best": best,
        "best_fitness_history": best_fitness_history,
        "best_conflicts_history": best_conflicts_history,
        "converged_at": converged_at,
        "final_conflicts": best.conflicts,
        "final_fitness": best.fitness,
        "generations_run": len(best_fitness_history),
        "runtime_s": runtime,
    }


# ----------------------------------------------------------------------------
# 3. Bộ thực nghiệm
# ----------------------------------------------------------------------------
N_RUNS_DEFAULT = 20
N_RUNS_SWEEP = 10
SEEDS = list(range(1000, 1000 + 50))


def run_experiments(rooms, classes):
    print("[1/3] Cấu hình mặc định pop=100, mut=0.1 ...")
    default_runs = []
    for s in SEEDS[:N_RUNS_DEFAULT]:
        r = run_ga(rooms, classes, s, population_size=100, mutation_rate=0.1)
        default_runs.append(r)
        print(f"   seed={s:4d}  conflicts={r['final_conflicts']:2d}  "
              f"converged@={r['converged_at']}  t={r['runtime_s']:.2f}s")

    print("[2/3] Khảo sát kích thước quần thể (50 / 100 / 200) ...")
    pop_sweep = {}
    for ps in [50, 100, 200]:
        runs = [run_ga(rooms, classes, s, population_size=ps, mutation_rate=0.1)
                for s in SEEDS[:N_RUNS_SWEEP]]
        pop_sweep[ps] = runs
        gens = [r["converged_at"] if r["converged_at"] is not None else 500
                for r in runs]
        print(f"   pop={ps:3d}  mean_gen={np.mean(gens):.1f}  "
              f"std={np.std(gens):.1f}")

    print("[3/3] Khảo sát mutation rate (0.05 / 0.10 / 0.20 / 0.30) ...")
    mut_sweep = {}
    for mr in [0.05, 0.10, 0.20, 0.30]:
        runs = [run_ga(rooms, classes, s, population_size=100, mutation_rate=mr)
                for s in SEEDS[:N_RUNS_SWEEP]]
        mut_sweep[mr] = runs
        gens = [r["converged_at"] if r["converged_at"] is not None else 500
                for r in runs]
        print(f"   mut={mr:.2f}  mean_gen={np.mean(gens):.1f}  "
              f"std={np.std(gens):.1f}")

    return default_runs, pop_sweep, mut_sweep


# ----------------------------------------------------------------------------
# 4. Vẽ biểu đồ
# ----------------------------------------------------------------------------
def pad_curve(curve, length, pad_val):
    """Kéo dài curve cho đủ length (sau khi GA đã dừng sớm)."""
    out = list(curve)
    if len(out) < length:
        out.extend([pad_val] * (length - len(out)))
    return out


def plot_convergence_default(default_runs, fname):
    max_len = max(len(r["best_conflicts_history"]) for r in default_runs)
    matrix = np.array([
        pad_curve(r["best_conflicts_history"], max_len, 0)
        for r in default_runs
    ])
    mean_curve = matrix.mean(axis=0)
    std_curve = matrix.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(max_len)
    for r in default_runs:
        c = pad_curve(r["best_conflicts_history"], max_len, 0)
        ax.plot(x, c, color="tab:blue", alpha=0.15, linewidth=1)
    ax.plot(x, mean_curve, color="tab:red", linewidth=2,
            label="Trung bình 20 lần chạy")
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                    color="tab:red", alpha=0.15, label="±1 độ lệch chuẩn")
    ax.set_xlabel("Thế hệ")
    ax.set_ylabel("Số xung đột (conflicts)")
    ax.set_title("Tiến trình hội tụ - cấu hình mặc định (pop=100, mut=0.1)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_sweep(sweep, label_fn, title, fname):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, runs in sweep.items():
        max_len = max(len(r["best_conflicts_history"]) for r in runs)
        matrix = np.array([
            pad_curve(r["best_conflicts_history"], max_len, 0) for r in runs
        ])
        mean_curve = matrix.mean(axis=0)
        ax.plot(np.arange(max_len), mean_curve, linewidth=2,
                label=label_fn(key))
    ax.set_xlabel("Thế hệ")
    ax.set_ylabel("Số xung đột (trung bình)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_boxplot(default_runs, fname):
    gens = [r["converged_at"] if r["converged_at"] is not None
            else r["generations_run"] for r in default_runs]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.boxplot(gens, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#A8D5BA"))
    ax.set_xticklabels(["pop=100, mut=0.1"])
    ax.set_ylabel("Thế hệ hội tụ (đạt 0 xung đột)")
    ax.set_title("Phân bố thế hệ hội tụ qua 20 lần chạy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. CSV tóm tắt + thống kê tóm tắt in ra console
# ----------------------------------------------------------------------------
def write_csv(out_path, default_runs, pop_sweep, mut_sweep):
    rows = []
    for i, r in enumerate(default_runs):
        rows.append(["default", "pop=100,mut=0.10", i, SEEDS[i],
                     r["final_conflicts"], r["final_fitness"],
                     r["converged_at"], r["generations_run"], r["runtime_s"]])
    for ps, runs in pop_sweep.items():
        for i, r in enumerate(runs):
            rows.append(["pop_sweep", f"pop={ps},mut=0.10", i, SEEDS[i],
                         r["final_conflicts"], r["final_fitness"],
                         r["converged_at"], r["generations_run"], r["runtime_s"]])
    for mr, runs in mut_sweep.items():
        for i, r in enumerate(runs):
            rows.append(["mut_sweep", f"pop=100,mut={mr:.2f}", i, SEEDS[i],
                         r["final_conflicts"], r["final_fitness"],
                         r["converged_at"], r["generations_run"], r["runtime_s"]])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "config", "run_idx", "seed",
                    "final_conflicts", "final_fitness",
                    "converged_at", "generations_run", "runtime_s"])
        w.writerows(rows)
    print(f"✓ Đã ghi {out_path}")


def print_summary(default_runs, pop_sweep, mut_sweep):
    """In các con số chính ra console - tiện copy vào báo cáo / LaTeX."""
    print("\n" + "=" * 70)
    print("TÓM TẮT THỐNG KÊ (copy vào báo cáo)")
    print("=" * 70)

    # Default
    final_conf = [r["final_conflicts"] for r in default_runs]
    success = sum(1 for c in final_conf if c == 0)
    gen_conv = [r["converged_at"] for r in default_runs
                if r["converged_at"] is not None]
    runtimes = [r["runtime_s"] for r in default_runs]
    fitness_final = [r["final_fitness"] for r in default_runs]

    print("\n[Default - pop=100, mut=0.10, 20 lần chạy]")
    print(f"  Thành công        : {success}/{len(default_runs)} "
          f"({100*success/len(default_runs):.0f}%)")
    print(f"  Fitness cuối      : {np.mean(fitness_final):.4f} "
          f"± {np.std(fitness_final):.4f}")
    print(f"  Conflicts cuối    : {np.mean(final_conf):.2f} "
          f"± {np.std(final_conf):.2f}")
    if gen_conv:
        print(f"  Gen hội tụ TB     : {np.mean(gen_conv):.2f} "
              f"± {np.std(gen_conv):.2f}")
        print(f"  Gen hội tụ min/max: {min(gen_conv)} / {max(gen_conv)}")
    print(f"  Runtime (s)       : {np.mean(runtimes):.3f} "
          f"± {np.std(runtimes):.3f}")

    # Pop sweep
    print("\n[Pop sweep - mut=0.10, 10 lần chạy / cấu hình]")
    print(f"  {'pop':>5}  {'success':>9}  {'gen_mean':>9}  "
          f"{'gen_std':>8}  {'time_s':>8}")
    for ps, runs in pop_sweep.items():
        gens = [r["converged_at"] if r["converged_at"] is not None else 500
                for r in runs]
        ok = sum(1 for r in runs if r["final_conflicts"] == 0)
        rts = [r["runtime_s"] for r in runs]
        print(f"  {ps:>5}  {ok:>2}/{len(runs):<6}  {np.mean(gens):>9.2f}  "
              f"{np.std(gens):>8.2f}  {np.mean(rts):>8.3f}")

    # Mut sweep
    print("\n[Mutation sweep - pop=100, 10 lần chạy / cấu hình]")
    print(f"  {'mut':>5}  {'success':>9}  {'gen_mean':>9}  "
          f"{'gen_std':>8}  {'time_s':>8}")
    for mr, runs in mut_sweep.items():
        gens = [r["converged_at"] if r["converged_at"] is not None else 500
                for r in runs]
        ok = sum(1 for r in runs if r["final_conflicts"] == 0)
        rts = [r["runtime_s"] for r in runs]
        print(f"  {mr:>5.2f}  {ok:>2}/{len(runs):<6}  {np.mean(gens):>9.2f}  "
              f"{np.std(gens):>8.2f}  {np.mean(rts):>8.3f}")
    print("=" * 70)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    rooms, classes = build_problem()
    default_runs, pop_sweep, mut_sweep = run_experiments(rooms, classes)

    plot_convergence_default(default_runs,
                             os.path.join(here, "fig_convergence_default.png"))
    plot_sweep(pop_sweep, lambda k: f"pop={k}",
               "Ảnh hưởng kích thước quần thể",
               os.path.join(here, "fig_convergence_population.png"))
    plot_sweep(mut_sweep, lambda k: f"mut={k:.2f}",
               "Ảnh hưởng tỷ lệ đột biến",
               os.path.join(here, "fig_convergence_mutation.png"))
    plot_boxplot(default_runs,
                 os.path.join(here, "fig_boxplot_convergence.png"))

    write_csv(os.path.join(here, "ga_runs_summary.csv"),
              default_runs, pop_sweep, mut_sweep)
    print_summary(default_runs, pop_sweep, mut_sweep)
