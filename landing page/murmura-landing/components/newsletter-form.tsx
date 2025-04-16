"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { subscribeToNewsletter } from "@/app/actions"
import { AlertCircle, CheckCircle, Loader2 } from "lucide-react"

export default function NewsletterForm() {
    const [email, setEmail] = useState("")
    const [isLoading, setIsLoading] = useState(false)
    const [message, setMessage] = useState<{ text: string; type: "success" | "error" } | null>(null)
    const [isValidEmail, setIsValidEmail] = useState(true)

    const validateEmail = (email: string) => {
        const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
        return regex.test(email)
    }

    const handleEmailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value
        setEmail(value)

        if (value) {
            setIsValidEmail(validateEmail(value))
        } else {
            setIsValidEmail(true) // Don't show error when field is empty
        }
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        // Validate email before submission
        if (!email) {
            setIsValidEmail(false)
            setMessage({ text: "Please enter your email address.", type: "error" })
            return
        }

        if (!validateEmail(email)) {
            setIsValidEmail(false)
            setMessage({ text: "Please enter a valid email address.", type: "error" })
            return
        }

        setIsLoading(true)
        setMessage(null)

        try {
            const formData = new FormData()
            formData.append("email", email)

            // In a real application, this would send an email to the specified address
            const result = await subscribeToNewsletter(formData)

            if (result.success) {
                setMessage({ text: result.message, type: "success" })
                setEmail("") // Clear the form on success
            } else {
                setMessage({ text: result.message, type: "error" })
            }
        } catch (error) {
            setMessage({
                text: "An error occurred. Please try again later.",
                type: "error",
            })
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="w-full">
            <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-2">
                <div className="relative flex-1">
                    <Input
                        type="email"
                        name="email"
                        placeholder="Enter your email"
                        value={email}
                        onChange={handleEmailChange}
                        className={`bg-white/10 text-white placeholder:text-white/50 border-white/20 focus-visible:ring-white ${
                            !isValidEmail ? "border-red-400" : ""
                        }`}
                        disabled={isLoading}
                        aria-invalid={!isValidEmail}
                    />
                </div>
                <Button type="submit" className="bg-white text-purple-600 hover:bg-white/90 min-w-[100px]" disabled={isLoading}>
                    {isLoading ? (
                        <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            <span>Sending...</span>
                        </>
                    ) : (
                        "Subscribe"
                    )}
                </Button>
            </form>

            {message && (
                <div
                    className={`mt-2 text-sm flex items-center ${message.type === "success" ? "text-green-100" : "text-red-200"}`}
                >
                    {message.type === "success" ? (
                        <CheckCircle className="h-4 w-4 mr-1" />
                    ) : (
                        <AlertCircle className="h-4 w-4 mr-1" />
                    )}
                    {message.text}
                </div>
            )}
        </div>
    )
}
