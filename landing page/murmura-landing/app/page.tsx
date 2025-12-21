import { Button } from "@/components/ui/button"
import { Network, Shield, Settings, BrainCircuit, Watch, Target, TrendingUp, AlertTriangle, CheckCircle, Database } from "lucide-react"
import Link from "next/link"
import NewsletterForm from "@/components/newsletter-form"
import { ReactNode } from 'react'

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
}

interface DatasetCardProps {
  name: string;
  source: string;
  nodes: string;
  activities: string;
  features: string;
}

interface AlgorithmCardProps {
  name: string;
  description: string;
  featured?: boolean;
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
              <Link href="/#insight" className="text-sm font-medium hover:text-primary">
                Key Insight
              </Link>
              <Link href="/#datasets" className="text-sm font-medium hover:text-primary">
                Datasets
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
                    <span className="mr-2 rounded-md bg-purple-600 px-2 py-0.5 text-xs text-white">Research</span>
                    <span className="text-muted-foreground">University of Melbourne</span>
                  </div>
                  <div className="space-y-2">
                    <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                      Evidential Trust-Aware Decentralized Federated Learning
                    </h1>
                    <p className="max-w-[600px] text-muted-foreground md:text-xl">
                      A modular framework for Byzantine-resilient decentralized FL with uncertainty-driven peer evaluation and personalized model aggregation for Wearable IoT.
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground italic">
                    Rangwala, Sinnott, Buyya - University of Melbourne
                  </p>
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
                      <Link href="https://doi.org/10.5281/zenodo.15622123" target="_blank" rel="noopener noreferrer">
                        <span className="flex items-center gap-2">
                          <Shield className="h-4 w-4" />
                          <span>DOI: 10.5281/zenodo.15622123</span>
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
                  Murmura is a comprehensive framework for <strong>Evidential Trust-Aware Decentralized Federated Learning</strong>.
                  It leverages Dirichlet-based uncertainty decomposition to evaluate peer trustworthiness and enable Byzantine-resilient
                  model aggregation for wearable IoT applications.
                </p>
                <div className="mt-6 flex flex-col md:flex-row gap-4 md:gap-8">
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <BrainCircuit className="h-12 w-12 text-purple-600" />
                    <h3 className="text-xl font-bold">Evidential Deep Learning</h3>
                    <p className="text-sm text-muted-foreground">
                      Dirichlet-based models for uncertainty quantification
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <Shield className="h-12 w-12 text-blue-600" />
                    <h3 className="text-xl font-bold">Byzantine Resilient</h3>
                    <p className="text-sm text-muted-foreground">
                      Robust against malicious nodes using trust-aware filtering
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <Watch className="h-12 w-12 text-indigo-600" />
                    <h3 className="text-xl font-bold">Wearable IoT</h3>
                    <p className="text-sm text-muted-foreground">
                      Designed for activity recognition on resource-constrained devices
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Key Insight Section */}
          <section id="insight" className="bg-gradient-to-br from-slate-800 to-slate-900 py-16 md:py-24 text-white">
            <div className="container px-4 md:px-6">
              <div className="mx-auto max-w-4xl text-center">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl mb-6">The Key Insight</h2>
                <p className="text-xl text-white/90 mb-8">
                  Epistemic-aleatoric uncertainty decomposition from Dirichlet-based evidential models directly indicates peer reliability:
                </p>
                <div className="bg-white/10 rounded-2xl p-8 space-y-6">
                  <div className="flex items-start gap-4 text-left">
                    <AlertTriangle className="text-amber-400 flex-shrink-0 mt-1 h-7 w-7" />
                    <div>
                      <h3 className="font-semibold text-lg mb-1">High Epistemic Uncertainty (Vacuity)</h3>
                      <p className="text-white/80">
                        Indicates insufficient learning or evidence - possibly Byzantine behavior. These peers should be filtered.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-4 text-left">
                    <CheckCircle className="text-green-400 flex-shrink-0 mt-1 h-7 w-7" />
                    <div>
                      <h3 className="font-semibold text-lg mb-1">High Aleatoric Uncertainty (Entropy)</h3>
                      <p className="text-white/80">
                        Reflects inherent data ambiguity - the peer is still trustworthy and should be included in aggregation.
                      </p>
                    </div>
                  </div>
                </div>
                <p className="mt-8 text-white/80">
                  This distinction allows intelligent peer filtering that traditional distance-based methods cannot achieve.
                </p>
              </div>
            </div>
          </section>

          {/* Features Section */}
          <section id="features" className="py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">Key Contributions</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Murmura introduces novel techniques for trust-aware decentralized federated learning.
                </p>
              </div>
              <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[64rem] md:grid-cols-3 lg:gap-8 mt-12">
                <FeatureCard
                    icon={<BrainCircuit className="h-10 w-10" />}
                    title="Evidential Trust-Aware Aggregation"
                    description="Novel algorithm leveraging Dirichlet-based uncertainty to identify and filter Byzantine peers via cross-evaluation."
                />
                <FeatureCard
                    icon={<Target className="h-10 w-10" />}
                    title="Uncertainty-Driven Personalization"
                    description="Adaptive self-weighting based on local model confidence, balancing knowledge transfer with personalized learning."
                />
                <FeatureCard
                    icon={<TrendingUp className="h-10 w-10" />}
                    title="BALANCE-Style Threshold Dynamics"
                    description="Progressive trust threshold tightening as models converge - starting lenient, becoming stricter over training."
                />
                <FeatureCard
                    icon={<Shield className="h-10 w-10" />}
                    title="Byzantine Attack Resilience"
                    description="Robust against Gaussian noise and directed deviation attacks with up to 30% compromised nodes."
                />
                <FeatureCard
                    icon={<Network className="h-10 w-10" />}
                    title="Flexible Topologies"
                    description="Support for ring, fully-connected, Erdos-Renyi, and k-regular network topologies."
                />
                <FeatureCard
                    icon={<Settings className="h-10 w-10" />}
                    title="Config-Driven Experiments"
                    description="YAML/JSON configuration for reproducible experiments with CLI and Python API support."
                />
              </div>
            </div>
          </section>

          {/* Datasets Section */}
          <section id="datasets" className="bg-slate-50 dark:bg-slate-900 py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center mb-12">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">Wearable IoT Datasets</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Evaluated on three real-world wearable sensor datasets with natural user heterogeneity.
                </p>
              </div>
              <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-3">
                <DatasetCard
                  name="UCI HAR"
                  source="Smartphone sensors"
                  nodes="10 subjects"
                  activities="6 classes"
                  features="561"
                />
                <DatasetCard
                  name="PAMAP2"
                  source="Body-worn IMUs"
                  nodes="9 subjects"
                  activities="12 classes"
                  features="4000"
                />
                <DatasetCard
                  name="PPG-DaLiA"
                  source="Wrist-worn PPG/EDA"
                  nodes="15 subjects"
                  activities="7 classes"
                  features="192"
                />
              </div>
            </div>
          </section>

          {/* Algorithms Section */}
          <section id="algorithms" className="py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center mb-12">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">Aggregation Algorithms</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Compare against state-of-the-art Byzantine-resilient aggregation methods.
                </p>
              </div>
              <div className="mx-auto grid max-w-4xl gap-4 grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
                <AlgorithmCard name="Evidential Trust" description="Uncertainty-aware" featured={true} />
                <AlgorithmCard name="FedAvg" description="Baseline averaging" />
                <AlgorithmCard name="Krum" description="Distance-based" />
                <AlgorithmCard name="BALANCE" description="Adaptive threshold" />
                <AlgorithmCard name="Sketchguard" description="Sketch compression" />
                <AlgorithmCard name="UBAR" description="Two-stage robust" />
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


function DatasetCard({ name, source, nodes, activities, features }: DatasetCardProps) {
  return (
    <div className="rounded-lg border bg-background p-6 shadow-sm">
      <div className="flex items-center gap-2 mb-4">
        <Database className="h-5 w-5 text-purple-600" />
        <h3 className="text-xl font-bold text-purple-600">{name}</h3>
      </div>
      <dl className="space-y-2 text-sm">
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Source</dt>
          <dd className="font-medium">{source}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Nodes</dt>
          <dd className="font-medium">{nodes}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Activities</dt>
          <dd className="font-medium">{activities}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Features</dt>
          <dd className="font-medium">{features}</dd>
        </div>
      </dl>
    </div>
  )
}

function AlgorithmCard({ name, description, featured = false }: AlgorithmCardProps) {
  return (
    <div className={`rounded-lg p-4 text-center transition-all hover:-translate-y-1 hover:shadow-lg ${
      featured
        ? 'bg-gradient-to-b from-purple-100 to-white dark:from-purple-900/30 dark:to-background border-2 border-purple-500'
        : 'bg-background border'
    }`}>
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

          {/* Animated trust pulse (behind central node) */}
          <circle cx="200" cy="200" r="45" fill="none" stroke="rgba(124, 58, 237, 0.4)" strokeWidth="2">
            <animate attributeName="r" from="45" to="80" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" from="0.8" to="0" dur="2s" repeatCount="indefinite" />
          </circle>

          {/* Central node with trust indicator (in front) */}
          <circle cx="200" cy="200" r="35" fill="rgba(124, 58, 237, 0.9)" />
          <text x="200" y="205" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Trust</text>

          {/* Honest nodes (green) */}
          <circle cx="100" cy="100" r="18" fill="rgba(34, 197, 94, 0.8)" />
          <circle cx="300" cy="120" r="18" fill="rgba(34, 197, 94, 0.8)" />
          <circle cx="80" cy="200" r="18" fill="rgba(34, 197, 94, 0.8)" />
          <circle cx="120" cy="300" r="18" fill="rgba(34, 197, 94, 0.8)" />
          <circle cx="320" cy="200" r="18" fill="rgba(34, 197, 94, 0.8)" />

          {/* Byzantine nodes (red, filtered) */}
          <circle cx="280" cy="300" r="18" fill="rgba(239, 68, 68, 0.6)" strokeDasharray="4,4" stroke="rgba(239, 68, 68, 0.8)" strokeWidth="2" />
        </svg>
      </div>
  )
}

