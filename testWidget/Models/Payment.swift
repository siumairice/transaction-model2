import SwiftUI

// Payment model to store our payment data
struct Payment: Identifiable {
    let id = UUID()
    let date: (month: String, day: String)
    let merchant: String
    let description: String
    let amount: String
    let paymentMethod: String
}

// Sample payment data
let samplePayments = [
    Payment(
        date: (month: "JUN", day: "30"),
        merchant: "Enbridge",
        description: "Bill Payment",
        amount: "52.19 USD",
        paymentMethod: "VISA (1234)"
    ),
    Payment(
        date: (month: "JUN", day: "28"),
        merchant: "Netflix",
        description: "Subscription",
        amount: "15.99 USD",
        paymentMethod: "VISA (1234)"
    ),
    Payment(
        date: (month: "JUN", day: "27"),
        merchant: "Whole Foods",
        description: "Groceries",
        amount: "87.32 USD",
        paymentMethod: "VISA (1234)"
    ),
    Payment(
        date: (month: "JUN", day: "25"),
        merchant: "Uber",
        description: "Transportation",
        amount: "24.50 USD",
        paymentMethod: "VISA (1234)"
    ),
    Payment(
        date: (month: "JUN", day: "23"),
        merchant: "AT&T",
        description: "Bill Payment",
        amount: "95.00 USD",
        paymentMethod: "VISA (1234)"
    )
] 