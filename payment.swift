import SwiftUI

// Transaction model to store our mock data
struct Payment: Identifiable {
    let id = UUID()
    let date: (month: String, day: String)
    let merchant: String
    let description: String
    let amount: String
    let paymentMethod: String
}

// Our payment row component, slightly modified to accept a Transaction
struct PaymentRow: View {
    let transaction: Payment
    
    var body: some View {
        HStack(spacing: 12) {
            // Date column with rounded square background
            VStack(alignment: .center, spacing: 0) {
                Text(transaction.date.month)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.secondary)
                Text(transaction.date.day)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.secondary)
            }
            .frame(width: 45, height: 45)
            .background(Color(.systemGray6))
            .cornerRadius(10)
            .padding(.leading, 2)
            
            // Main content
            VStack(alignment: .leading, spacing: 4) {
                Text(transaction.merchant)
                    .font(.system(size: 17, weight: .semibold))
                Text(transaction.description)
                    .font(.system(size: 15))
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Amount and payment method
            VStack(alignment: .trailing, spacing: 4) {
                Text(transaction.amount)
                    .font(.system(size: 17, weight: .semibold))
                Text(transaction.paymentMethod)
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)
            }
            .padding(.trailing, 4)
        }
        .padding(.vertical, 12)
        .background(Color(.systemBackground))
    }
}

// View to display the list of transactions
struct PaymentListView: View {
    // Sample mock data
    let transactions = [
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
        ),
        Payment(
            date: (month: "JUN", day: "21"),
            merchant: "Starbucks",
            description: "Coffee Shop",
            amount: "6.75 USD",
            paymentMethod: "VISA (1234)"
        ),
        Payment(
            date: (month: "JUN", day: "20"),
            merchant: "Amazon",
            description: "Online Shopping",
            amount: "42.99 USD",
            paymentMethod: "VISA (1234)"
        )
    ]
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Recent Transactions")) {
                    ForEach(transactions) { payment in
                        PaymentRow(transaction: payment)
                            .listRowInsets(EdgeInsets(top: 0, leading: 0, bottom: 0, trailing: 0))
                            .listRowSeparator(.hidden)
                    }
                }
            }
            .listStyle(InsetGroupedListStyle())
            .navigationTitle("Transaction History")
        }
    }
}

struct TransactionListView_Previews: PreviewProvider {
    static var previews: some View {
        PaymentListView()
    }
}
