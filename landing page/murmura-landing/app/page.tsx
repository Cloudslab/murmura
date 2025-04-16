import { Button } from "@/components/ui/button"
// Import icons from lucide-react
import { Network, Shield, Code2, Settings, BarChart3, BrainCircuit, Atom, Check } from "lucide-react"
// Import SVG component from next/image
import Link from "next/link"
import NewsletterForm from "@/components/newsletter-form"
import { ReactNode } from 'react'

// Define TypeScript interfaces for component props
interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
}

interface StatusItemProps {
  text: string;
  status: 'completed' | 'in-progress' | 'planned';
}

interface RoadmapItemProps {
  title: string;
  description: string;
  icon: ReactNode;
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
              <Link href="/#status" className="text-sm font-medium hover:text-primary">
                Status
              </Link>
              <Link href="/#roadmap" className="text-sm font-medium hover:text-primary">
                Roadmap
              </Link>
            </nav>
            <div>
              <Button variant="outline" className="flex items-center gap-2" asChild>
                <Link href="https://github.com/murtazahr/murmura" target="_blank" rel="noopener noreferrer">
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
                    <span className="mr-2 rounded-md bg-purple-600 px-2 py-0.5 text-xs text-white">New</span>
                    <span className="text-muted-foreground">Under active development</span>
                  </div>
                  <div className="space-y-2">
                    <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                      Decentralized Learning Framework
                    </h1>
                    <p className="max-w-[600px] text-muted-foreground md:text-xl">
                      Empowering researchers and developers to experiment with distributed machine learning while
                      maintaining privacy.
                    </p>
                  </div>
                  <div className="flex flex-col gap-2 min-[400px]:flex-row">
                    <Button className="bg-gradient-to-r from-blue-600 to-purple-600 text-white" asChild>
                      <Link href="https://github.com/murtazahr/murmura" target="_blank" rel="noopener noreferrer">
                        <span className="flex items-center gap-2">
                          <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                          </svg>
                          <span>GitHub Repository</span>
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
                  Murmura is a framework for federated learning in decentralized/P2P environments. It helps researchers
                  and developers experiment with distributed machine learning while maintaining privacy.
                </p>
                <div className="mt-6 flex flex-col md:flex-row gap-4 md:gap-8">
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <Network className="h-12 w-12 text-purple-600" />
                    <h3 className="text-xl font-bold">Decentralized</h3>
                    <p className="text-sm text-muted-foreground">
                      Built for peer-to-peer environments with no central authority
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <Shield className="h-12 w-12 text-blue-600" />
                    <h3 className="text-xl font-bold">Privacy-Preserving</h3>
                    <p className="text-sm text-muted-foreground">
                      Keeps data private while enabling collaborative learning
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-2 rounded-lg border bg-background p-6 text-center shadow-sm">
                    <Code2 className="h-12 w-12 text-indigo-600" />
                    <h3 className="text-xl font-bold">Experimental</h3>
                    <p className="text-sm text-muted-foreground">
                      Designed for research and development of new ML techniques
                    </p>
                  </div>
                </div>
                <div className="mt-8 inline-flex items-center rounded-lg bg-blue-100 dark:bg-blue-900/30 px-4 py-2 text-sm">
                  <span className="mr-2 rounded-md bg-blue-600 px-2 py-0.5 text-xs text-white">Note</span>
                  <span className="text-blue-800 dark:text-blue-200">Murmura is currently under active development</span>
                </div>
              </div>
            </div>
          </section>

          {/* Features Section */}
          <section id="features" className="py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">Key Features</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Murmura provides a comprehensive toolkit for decentralized federated learning research and development.
                </p>
              </div>
              <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[64rem] md:grid-cols-3 lg:gap-8 mt-12">
                <FeatureCard
                    icon={<Network className="h-10 w-10" />}
                    title="Decentralized Federated Learning"
                    description="Implement federated learning algorithms in fully decentralized environments without central coordination."
                />
                <FeatureCard
                    icon={<Settings className="h-10 w-10" />}
                    title="P2P Network Simulation"
                    description="Simulate various network conditions and topologies to test algorithm robustness."
                />
                <FeatureCard
                    icon={<Shield className="h-10 w-10" />}
                    title="Privacy-Preserving Protocols"
                    description="Implement and test various privacy-preserving learning protocols and techniques."
                />
                <FeatureCard
                    icon={<Code2 className="h-10 w-10" />}
                    title="Customizable Node Behaviors"
                    description="Define custom node behaviors and network topologies for diverse experimental setups."
                />
                <FeatureCard
                    icon={<BarChart3 className="h-10 w-10" />}
                    title="Performance Metrics"
                    description="Comprehensive metrics and analysis tools to evaluate algorithm performance."
                />
                <FeatureCard
                    icon={<BrainCircuit className="h-10 w-10" />}
                    title="Model Integration"
                    description="Easily integrate with popular machine learning frameworks and models."
                />
              </div>
            </div>
          </section>

          {/* Development Status Section */}
          <section id="status" className="bg-slate-50 dark:bg-slate-900 py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto grid max-w-5xl items-center gap-6 lg:grid-cols-2 lg:gap-12">
                <div className="space-y-4">
                  <div className="inline-flex items-center rounded-lg bg-muted px-3 py-1 text-sm">
                    <span className="text-purple-600 font-medium">Development Status</span>
                  </div>
                  <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">Alpha Release 1</h2>
                  <p className="text-muted-foreground md:text-lg">
                    Murmura is currently in active development with key components being implemented and tested for our
                    first alpha release. Our team is working diligently to create a robust framework for decentralized
                    learning research.
                  </p>
                  <ul className="space-y-2">
                    <StatusItem text="Core P2P networking layer" status="completed" />
                    <StatusItem text="Basic federated learning algorithms" status="completed" />
                    <StatusItem text="Privacy-preserving protocols" status="in-progress" />
                    <StatusItem text="Performance metrics dashboard" status="in-progress" />
                    <StatusItem text="Advanced network simulation" status="planned" />
                    <StatusItem text="Documentation and examples" status="planned" />
                  </ul>
                </div>
                <div className="flex justify-center">
                  <div className="relative h-[300px] w-[300px] md:h-[400px] md:w-[400px]">
                    <DevelopmentGraphic />
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Roadmap Section */}
          <section id="roadmap" className="py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">Future Roadmap</h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Our vision for Murmura extends beyond current capabilities. Here&apos;s what we&apos;re planning for the future.
                </p>
              </div>
              <div className="mx-auto mt-12 max-w-3xl">
                <div className="space-y-8">
                  <RoadmapItem
                      title="AI Agent Integration"
                      description="Integrate AI agents within the framework to enable autonomous learning and adaptation in decentralized environments."
                      icon={<BrainCircuit className="h-8 w-8 text-purple-600" />}
                  />
                  <RoadmapItem
                      title="Quantum Node Emulation"
                      description="Develop quantum node emulation capabilities for experimenting with quantum federated learning approaches."
                      icon={<Atom className="h-8 w-8 text-blue-600" />}
                  />
                  <RoadmapItem
                      title="Advanced Privacy Techniques"
                      description="Implement cutting-edge privacy-preserving techniques including homomorphic encryption and secure multi-party computation."
                      icon={<Shield className="h-8 w-8 text-indigo-600" />}
                  />
                  <RoadmapItem
                      title="Real-world Deployment Tools"
                      description="Create tools and frameworks for deploying Murmura-based systems in real-world decentralized environments."
                      icon={<Network className="h-8 w-8 text-blue-600" />}
                  />
                </div>
              </div>
            </div>
          </section>

          {/* Newsletter Section */}
          <section className="bg-gradient-to-r from-blue-600 to-purple-600 py-16 md:py-24">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <h2 className="text-3xl font-bold leading-[1.1] text-white sm:text-3xl md:text-5xl">Stay Updated</h2>
                <p className="max-w-[85%] leading-normal text-white/80 sm:text-lg sm:leading-7">
                  Subscribe to our newsletter to receive updates on Murmura&apos;s development and be the first to know about
                  new features and releases.
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
                  href="https://github.com/murtazahr/murmura"
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
                  We respect your privacy. This project does not collect any personal data. Any information provided
                  through contact forms is used solely for communication purposes.
                </div>
              </Link>
            </nav>
            <p className="text-sm text-muted-foreground">© 2025 Murmura. All rights reserved.</p>
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

function StatusItem({ text, status }: StatusItemProps) {
  return (
      <li className="flex items-center gap-2">
        <div
            className={`h-2 w-2 rounded-full ${
                status === "completed" ? "bg-green-500" : status === "in-progress" ? "bg-yellow-500" : "bg-blue-500"
            }`}
        />
        <span className="text-sm">{text}</span>
        <span
            className={`ml-auto text-xs ${
                status === "completed" ? "text-green-500" : status === "in-progress" ? "text-yellow-500" : "text-blue-500"
            }`}
        >
        {status === "completed" ? "Completed" : status === "in-progress" ? "In Progress" : "Planned"}
      </span>
      </li>
  )
}

function RoadmapItem({ title, description, icon }: RoadmapItemProps) {
  return (
      <div className="flex gap-4">
        <div className="flex-shrink-0 mt-1">{icon}</div>
        <div className="space-y-1">
          <h3 className="text-xl font-bold">{title}</h3>
          <p className="text-muted-foreground">{description}</p>
        </div>
      </div>
  )
}

function NetworkGraphic() {
  return (
      <div className="relative h-full w-full">
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 400 400">
          {/* Central node */}
          <circle cx="200" cy="200" r="30" fill="rgba(124, 58, 237, 0.8)" />

          {/* Outer nodes */}
          <circle cx="100" cy="100" r="15" fill="rgba(79, 70, 229, 0.8)" />
          <circle cx="300" cy="120" r="15" fill="rgba(124, 58, 237, 0.8)" />
          <circle cx="120" cy="280" r="15" fill="rgba(79, 70, 229, 0.8)" />
          <circle cx="280" cy="280" r="15" fill="rgba(124, 58, 237, 0.8)" />
          <circle cx="80" cy="200" r="15" fill="rgba(79, 70, 229, 0.8)" />
          <circle cx="320" cy="200" r="15" fill="rgba(124, 58, 237, 0.8)" />

          {/* Connection lines */}
          <line x1="200" y1="200" x2="100" y2="100" stroke="rgba(124, 58, 237, 0.5)" strokeWidth="2" />
          <line x1="200" y1="200" x2="300" y2="120" stroke="rgba(124, 58, 237, 0.5)" strokeWidth="2" />
          <line x1="200" y1="200" x2="120" y2="280" stroke="rgba(124, 58, 237, 0.5)" strokeWidth="2" />
          <line x1="200" y1="200" x2="280" y2="280" stroke="rgba(124, 58, 237, 0.5)" strokeWidth="2" />
          <line x1="200" y1="200" x2="80" y2="200" stroke="rgba(124, 58, 237, 0.5)" strokeWidth="2" />
          <line x1="200" y1="200" x2="320" y2="200" stroke="rgba(124, 58, 237, 0.5)" strokeWidth="2" />

          {/* Secondary connections */}
          <line
              x1="100"
              y1="100"
              x2="80"
              y2="200"
              stroke="rgba(124, 58, 237, 0.3)"
              strokeWidth="1"
              strokeDasharray="5,5"
          />
          <line
              x1="300"
              y1="120"
              x2="320"
              y2="200"
              stroke="rgba(124, 58, 237, 0.3)"
              strokeWidth="1"
              strokeDasharray="5,5"
          />
          <line
              x1="120"
              y1="280"
              x2="280"
              y2="280"
              stroke="rgba(124, 58, 237, 0.3)"
              strokeWidth="1"
              strokeDasharray="5,5"
          />

          {/* Animated pulses */}
          <circle cx="200" cy="200" r="40" fill="none" stroke="rgba(124, 58, 237, 0.3)" strokeWidth="2">
            <animate attributeName="r" from="40" to="70" dur="3s" repeatCount="indefinite" />
            <animate attributeName="opacity" from="0.8" to="0" dur="3s" repeatCount="indefinite" />
          </circle>
        </svg>
      </div>
  )
}

function DevelopmentGraphic() {
  return (
      <div className="relative h-full w-full">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="grid grid-cols-3 gap-4">
            {[...Array(9)].map((_, i) => (
                <div
                    key={i}
                    className={`h-16 w-16 rounded-lg ${
                        i < 3 ? "bg-green-500/80" : i < 6 ? "bg-yellow-500/80" : "bg-blue-500/30"
                    } flex items-center justify-center text-white font-bold`}
                >
                  {i < 3 ? (
                      <Check className="h-8 w-8" />
                  ) : i < 6 ? (
                      <div className="h-2 w-8 bg-white rounded-full animate-pulse" />
                  ) : (
                      ""
                  )}
                </div>
            ))}
          </div>
        </div>
      </div>
  )
}