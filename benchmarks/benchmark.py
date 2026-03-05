import subprocess
import re
import matplotlib.pyplot as plt

# Configuration
# Replace these with the actual paths to your compiled executables
EXECUTABLES = {
    "Sequential": "../sequential/main", 
    "OpenMP": "../openMP/main",
    "CUDA": "../cuda/main"
}

# The particle counts you want to test
PARTICLE_COUNTS = [100, 500, 1000, 2000, 5000]

def run_benchmark(executable, n_particles):
    """Runs a single simulation and extracts the Average FPS."""
    print(f"Running {executable} with N={n_particles}...")
    try:
        # Run the executable and pass N as a command line argument
        result = subprocess.run(
            [executable, str(n_particles)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output for "Average FPS: <value>"
        match = re.search(r"Average FPS:\s*([0-9.]+)", result.stdout)
        if match:
            return float(match.group(1))
        else:
            print(f"  Warning: Could not parse FPS from {executable} output.")
            return 0.0

    except subprocess.CalledProcessError as e:
        print(f"  Error running {executable}: {e}")
        return 0.0
    except FileNotFoundError:
        print(f"  Executable {executable} not found. Skipping.")
        return 0.0

def main():
    results = {name: [] for name in EXECUTABLES.keys()}

    # Run tests
    for n in PARTICLE_COUNTS:
        for name, exe in EXECUTABLES.items():
            fps = run_benchmark(exe, n)
            results[name].append(fps)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for (name, fps_list), marker, color in zip(results.items(), markers, colors):
        plt.plot(PARTICLE_COUNTS, fps_list, marker=marker, color=color, label=name, linewidth=2)

    plt.title('N-Body Simulation Performance: Average FPS vs Particle Count', fontsize=14)
    plt.xlabel('Number of Particles (N)', fontsize=12)
    plt.ylabel('Average Frames Per Second (FPS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('benchmark_results.png')
    print("Benchmarking complete. Results saved to benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    main()
