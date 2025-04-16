"use server"

import { Resend } from 'resend';

export async function subscribeToNewsletter(formData: FormData) {
    const email = formData.get("email") as string

    if (!email || !email.includes("@")) {
        return {
            success: false,
            message: "Please enter a valid email address.",
        }
    }

    try {
        // Initialize the Resend client with your API key from environment variables
        const resend = new Resend(process.env.RESEND_API_KEY);

        // Send confirmation email to the new subscriber
        const {error } = await resend.emails.send({
            from: 'Murmura Team <newsletter@murtaza-hatim.com>',
            to: [email],
            replyTo: ['mrangwala@student.unimelb.edu.au'],
            subject: 'Welcome to Murmura\'s Journey',
            html: `
                <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 0 auto; padding: 30px; border-radius: 8px; border: 1px solid #eaeaea; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h1 style="color: #333; margin-bottom: 5px; font-size: 24px;">Welcome to Murmura</h1>
                        <p style="color: #666; font-size: 16px; margin-top: 0;">Thank you for joining our community</p>
                    </div>
                    
                    <div style="color: #444; font-size: 15px; line-height: 1.6; margin: 25px 0;">
                        <p>Thank you for your interest in Murmura. We're delighted to have you with us on this journey.</p>
                        <p>We're working diligently on our alpha release and will keep you informed of our progress and upcoming features. Your early support means a great deal to us.</p>
                        <p>If you have any questions or feedback, please don't hesitate to reach out by replying directly to this email.</p>
                    </div>
                    
                    <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #888; text-align: center;">
                        <p>© ${new Date().getFullYear()} Murmura. All rights reserved.</p>
                        <p>If you didn't sign up for this newsletter, you can safely ignore this email.</p>
                    </div>
                </div>
            `
        });

        if (error) {
            console.error("Resend API error:", error);
            return {
                success: false,
                message: "We couldn't send the confirmation email. Please try again later.",
            }
        }

        // Only proceed with notification email if first email was successful
        try {
            // Send notification to yourself about the new subscriber
            await resend.emails.send({
                from: 'Murmura Subscriptions <newsletter@murtaza-hatim.com>',
                to: ['mrangwala@student.unimelb.edu.au'],
                subject: 'New Murmura Subscriber',
                html: `
                    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px;">
                        <h2 style="color: #333;">New Subscriber Alert</h2>
                        <p style="font-size: 16px;">A new user has subscribed to the Murmura newsletter:</p>
                        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 4px; margin: 15px 0;">
                            <p style="margin: 0;"><strong>Email:</strong> ${email}</p>
                            <p style="margin: 10px 0 0;"><strong>Date:</strong> ${new Date().toLocaleString()}</p>
                        </div>
                    </div>
                `
            });
        } catch (notificationError) {
            // Log notification error but don't affect user experience
            console.error("Error sending notification email:", notificationError);
            // We still return success to the user since their subscription was processed
        }

        return {
            success: true,
            message: "Thank you for subscribing! Check your inbox for a confirmation.",
        }
    } catch (error) {
        console.error("Error subscribing to newsletter:", error)
        return {
            success: false,
            message: "We couldn't process your subscription. Please try again later.",
        }
    }
}