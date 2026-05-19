import { Button } from "@/components/ui/button"
import { Network, Shield, Settings, Layers, GitBranch, Terminal, Cpu, Package } from "lucide-react"
import Link from "next/link"
import NewsletterForm from "@/components/newsletter-form"
import { ReactNode } from 'react'

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
}

interface AlgorithmCardProps {
  name: string;
  description: string;
}

export default function Home() {
  return (
      <div className="flex min-h-screen flex-col">
        {/* Header */}
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <Link href="/" className="flex items-center gap-2">
                <Network className="h-6 w-6 text-purple-600" />
                <span className="text-xl font-bold tracking-tight">Murmura</span>
              </Link>
            </div>
            <nav className="hidden md:flex gap-6">
              <Link href="/#overview" className="text-sm font-medium hover:text-primary">
                Overview
              </Link>
              <Link href="/#features" className="text-sm font-medium hover:text-primary">
                Features
              </Link>
              <Link href="/#quickstart" className="text-sm font-medium hover:text-primary">
                Quick Start
              </Link>
              <Link href="/#algorithms" className="text-sm font-medium hover:text-primary">
                Algorithms
              </Link>
            </nav>
            <div>
              <Button variant="outline" className="flex items-center gap-2" asChild>
                <Link href="https://github.com/Cloudslab/murmura" target="_blank" rel="noopener noreferrer">
                  <span className="flex items-center gap-2">
                    <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                    </svg>
                    <span>GitHub</span>
                  </span>
                </Link>
              </Button>
            </div>
          </div>
        </header>

        <main className="flex-1">
          {/* Hero Section */}
          <section className="relative py-20 md:py-28">
            <div className="container px-4 md:px-6">
              <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
                <div className="flex flex-col justify-center space-y-4">
                  <div className="inline-flex items-center rounded-lg bg-muted px-3 py-1 text-sm max-w-fit">
                    <span className="mr-2 rounded-md bg-purple-600 px-2 py-0.5 text-xs text-white">Open Source</span>
                    <span className="text-muted-foreground">University of Melbourne</span>
                  </div>
                  <div className="space-y-2">
                    <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                      Decentralized Federated Learning, Made Modular
                    </h1>
                    <p className="max-w-[600px] text-muted-foreground md:text-xl">
                      Murmura is an open-source framework for running Byzantine-resilient decentralized federated learning experiments. Swap aggregators, topologies, and backends from a single config file.
                    </p>
                  </div>
                  <div className="flex flex-col gap-2 min-[400px]:flex-row">
                    <Button className="bg-gradient-to-r from-blue-600 to-purple-600 text-white" asChild>
                      <Link href="https://github.com/Cloudslab/murmura" target="_blank" rel="noopener noreferrer">
                        <span className="flex items-center gap-2">
                          <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                          </svg>
                          <span>GitHub Repository</span>
                        </span>
                      </Link>
                    </Button>
                    <Button variant="outline" asChild>
                      <Link href="/#quickstart">
                        <span className="flex items-center gap-2">
                          <Terminal className="h-4 w-4" />
                          <span>Quick Start</span>
                        </span>
                      </Link>
                    </Button>
                  </div>
                </div>
                <div className="flex items-center justify-center">
                  <div className="relative h-[300px] w-[300px] md:h-[400px] md:w-[400px]">
                    <NetworkGraphic />
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Overview Section */}
          <section id="overview" className="bg-slate-50 dark:bg-slate-900 py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">What is Murmura?</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Murmura is a config-driven research framework for decentralized federated learning. Define your
                  topology, aggregation algorithm, attack scenario, and execution backend in a single YAML file
                  and run reproducible experiments with one command.
                </p>
                <div className="mt-6 flex flex-col md:flex-row gap-4 md:gap-8">
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <Settings className="h-12 w-12 text-purple-600" />
                    <h3 className="text-xl font-bold">Config-Driven</h3>
                    <p className="text-sm text-muted-foreground">
                      Full experiments defined in YAML — reproducible, shareable, version-controlled
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <Layers className="h-12 w-12 text-blue-600" />
                    <h3 className="text-xl font-bold">Modular</h3>
                    <p className="text-sm text-muted-foreground">
                      Plug in any aggregator, topology, or attack without touching the training loop
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <Cpu className="h-12 w-12 text-indigo-600" />
                    <h3 className="text-xl font-bold">Two Backends</h3>
                    <p className="text-sm text-muted-foreground">
                      Fast in-process simulation or real distributed ZMQ processes — same config
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Features Section */}
          <section id="features" className="py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">Framework Features</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Everything you need to design, run, and extend decentralized FL experiments.
                </p>
              </div>
              <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[64rem] md:grid-cols-3 lg:gap-8 mt-12">
                <FeatureCard
                    icon={<Network className="h-10 w-10" />}
                    title="Flexible Topologies"
                    description="Ring, fully-connected, Erdős-Rényi, and k-regular graphs. Topology is decoupled from training logic."
                />
                <FeatureCard
                    icon={<Shield className="h-10 w-10" />}
                    title="Byzantine Attack Simulation"
                    description="Built-in Gaussian noise, directed deviation, and topology-liar attacks with configurable Byzantine fraction."
                />
                <FeatureCard
                    icon={<Cpu className="h-10 w-10" />}
                    title="Simulation & Distributed Backends"
                    description="Run in-process for fast iteration or as real OS processes over ZeroMQ — switch with one config line."
                />
                <FeatureCard
                    icon={<GitBranch className="h-10 w-10" />}
                    title="Dynamic Topology Support"
                    description="Mobility model generates time-varying graphs deterministically from a shared seed across all nodes."
                />
                <FeatureCard
                    icon={<Settings className="h-10 w-10" />}
                    title="Config-Driven Experiments"
                    description="YAML or JSON configs with Pydantic validation. Every experiment is fully reproducible and shareable."
                />
                <FeatureCard
                    icon={<Package className="h-10 w-10" />}
                    title="Extensible by Design"
                    description="Add a new aggregator, topology, or attack by subclassing a single base class — no framework internals to touch."
                />
              </div>
            </div>
          </section>

          {/* Quick Start Section */}
          <section id="quickstart" className="bg-slate-50 dark:bg-slate-900 py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center mb-10">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">Quick Start</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Install, configure, and run your first experiment in minutes.
                </p>
              </div>
              <div className="mx-auto max-w-3xl space-y-6">
                <div className="rounded-lg border bg-background shadow-sm overflow-hidden">
                  <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-slate-200 text-sm font-mono">
                    <Terminal className="h-4 w-4" />
                    <span>Install</span>
                  </div>
                  <pre className="p-4 text-sm font-mono overflow-x-auto bg-slate-900 text-green-400">
{`pip install murmura
# or with uv (recommended)
uv pip install murmura`}
                  </pre>
                </div>
                <div className="rounded-lg border bg-background shadow-sm overflow-hidden">
                  <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-slate-200 text-sm font-mono">
                    <Terminal className="h-4 w-4" />
                    <span>Run an experiment</span>
                  </div>
                  <pre className="p-4 text-sm font-mono overflow-x-auto bg-slate-900 text-green-400">
{`# Run with a config file
murmura run experiments/basic_fedavg.yaml

# List available components
murmura list-components aggregators
murmura list-components topologies
murmura list-components attacks`}
                  </pre>
                </div>
                <div className="rounded-lg border bg-background shadow-sm overflow-hidden">
                  <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-slate-200 text-sm font-mono">
                    <Settings className="h-4 w-4" />
                    <span>Example config (YAML)</span>
                  </div>
                  <pre className="p-4 text-sm font-mono overflow-x-auto bg-slate-900 text-slate-200">
{`experiment:
  name: "my-experiment"
  rounds: 50

topology:
  type: "k-regular"
  num_nodes: 20
  k: 4

aggregation:
  algorithm: "krum"

attack:
  enabled: true
  type: "gaussian"
  percentage: 0.2

backend: simulation`}
                  </pre>
                </div>
              </div>
            </div>
          </section>

          {/* Algorithms Section */}
          <section id="algorithms" className="py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center mb-12">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">Built-in Aggregators</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Six aggregation algorithms ready to use out of the box — or bring your own by subclassing <code className="text-sm bg-muted px-1 py-0.5 rounded">Aggregator</code>.
                </p>
              </div>
              <div className="mx-auto grid max-w-4xl gap-4 grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
                <AlgorithmCard name="FedAvg" description="Simple averaging" />
                <AlgorithmCard name="Krum" description="Distance-based filtering" />
                <AlgorithmCard name="BALANCE" description="Adaptive threshold" />
                <AlgorithmCard name="Sketchguard" description="Sketch compression" />
                <AlgorithmCard name="UBAR" description="Two-stage robust" />
                <AlgorithmCard name="Evidential Trust" description="Uncertainty-aware" />
              </div>
            </div>
          </section>

          {/* Newsletter Section */}
          <section className="bg-gradient-to-r from-blue-600 to-purple-600 py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] text-white sm:text-3xl md:text-5xl">Stay Updated</h2>
                <p className="max-w-[85%] leading-normal text-white/80 sm:text-lg sm:leading-7">
                  Subscribe to receive updates on Murmura&apos;s development and new releases.
                </p>
                <div className="w-full max-w-md space-y-2">
                  <NewsletterForm />
                  <p className="text-xs text-white/60">We respect your privacy. Unsubscribe at any time.</p>
                </div>
              </div>
            </div>
          </section>
        </main>

        {/* Footer */}
        <footer className="border-t bg-background">
          <div className="container flex flex-col gap-6 py-8 md:flex-row md:items-center md:justify-between md:py-12">
            <Link href="/" className="flex items-center gap-2">
              <Network className="h-6 w-6 text-purple-600" />
              <span className="text-lg font-bold">Murmura</span>
            </Link>
            <nav className="flex gap-4 md:gap-6">
              <Link
                  href="https://github.com/Cloudslab/murmura"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-medium hover:underline underline-offset-4"
              >
                GitHub
              </Link>
              <Link
                  href="mailto:mrangwala@student.unimelb.edu.au"
                  className="text-sm font-medium hover:underline underline-offset-4"
              >
                Contact
              </Link>
              <Link href="#" className="text-sm font-medium hover:underline underline-offset-4 group relative">
                Privacy
                <div className="absolute bottom-full mb-2 left-0 w-64 p-2 bg-background border rounded-md shadow-md hidden group-hover:block text-xs text-muted-foreground z-50">
                  We respect your privacy. This project does not collect any personal data.
                </div>
              </Link>
            </nav>
            <p className="text-sm text-muted-foreground">&copy; 2025 Murmura - University of Melbourne</p>
          </div>
        </footer>
      </div>
  )
}

function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
      <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm transition-all hover:shadow-md">
        <div className="text-purple-600">{icon}</div>
        <h3 className="text-xl font-bold">{title}</h3>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
  )
}

function AlgorithmCard({ name, description }: AlgorithmCardProps) {
  return (
    <div className="rounded-lg p-4 text-center bg-background border transition-all hover:-translate-y-1 hover:shadow-lg">
      <h4 className="font-semibold mb-1">{name}</h4>
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  )
}

function NetworkGraphic() {
  return (
      <div className="relative h-full w-full">
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 400 400">
          {/* Peer connections (background) */}
          <line x1="100" y1="100" x2="80" y2="200" stroke="rgba(124, 58, 237, 0.3)" strokeWidth="1" />
          <line x1="300" y1="120" x2="320" y2="200" stroke="rgba(124, 58, 237, 0.3)" strokeWidth="1" />
          <line x1="120" y1="300" x2="80" y2="200" stroke="rgba(124, 58, 237, 0.3)" strokeWidth="1" />

          {/* Trust connections (solid for trusted) */}
          <line x1="200" y1="200" x2="100" y2="100" stroke="rgba(34, 197, 94, 0.6)" strokeWidth="3" />
          <line x1="200" y1="200" x2="300" y2="120" stroke="rgba(34, 197, 94, 0.6)" strokeWidth="3" />
          <line x1="200" y1="200" x2="80" y2="200" stroke="rgba(34, 197, 94, 0.6)" strokeWidth="3" />
          <line x1="200" y1="200" x2="120" y2="300" stroke="rgba(34, 197, 94, 0.6)" strokeWidth="3" />
          <line x1="200" y1="200" x2="320" y2="200" stroke="rgba(34, 197, 94, 0.6)" strokeWidth="3" />

          {/* Filtered connection (dashed red) */}
          <line x1="200" y1="200" x2="280" y2="300" stroke="rgba(239, 68, 68, 0.5)" strokeWidth="2" strokeDasharray="8,4" />

          {/* Animated trust pulse */}
          <circle cx="200" cy="200" r="45" fill="none" stroke="rgba(124, 58, 237, 0.4)" strokeWidth="2">
            <animate attributeName="r" from="45" to="80" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" from="0.8" to="0" dur="2s" repeatCount="indefinite" />
          </circle>

          {/* Central node */}
          <circle cx="200" cy="200" r="35" fill="rgba(124, 58, 237, 0.9)" />
          <text x="200" y="205" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Node</text>

          {/* Honest nodes (green) */}
          <circle cx="100" cy="100" r="18" fill="rgba(34, 197, 94, 0.8)" />
          <circle cx="300" cy="120" r="18" fill="rgba(34, 197, 94, 0.8)" />
          <circle cx="80" cy="200" r="18" fill="rgba(34, 197, 94, 0.8)" />
          <circle cx="120" cy="300" r="18" fill="rgba(34, 197, 94, 0.8)" />
          <circle cx="320" cy="200" r="18" fill="rgba(34, 197, 94, 0.8)" />

          {/* Byzantine node (red, filtered) */}
          <circle cx="280" cy="300" r="18" fill="rgba(239, 68, 68, 0.6)" strokeDasharray="4,4" stroke="rgba(239, 68, 68, 0.8)" strokeWidth="2" />
        </svg>
      </div>
  )
}
