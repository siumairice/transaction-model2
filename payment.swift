import SwiftUI
import WidgetKit

// Payment model to store our mock data
struct Payment: Identifiable {
    let id = UUID()
    let date: (month: String, day: String)
    let merchant: String
    let description: String
    let amount: String
    let paymentMethod: String
}

// Provider for widget data
struct Provider: TimelineProvider {
    func placeholder(in context: Context) -> PaymentEntry {
        PaymentEntry(date: Date(), payments: samplePayments)
    }

    func getSnapshot(in context: Context, completion: @escaping (PaymentEntry) -> ()) {
        let entry = PaymentEntry(date: Date(), payments: samplePayments)
        completion(entry)
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<PaymentEntry>) -> ()) {
        // Update once per day
        var entries: [PaymentEntry] = []
        let currentDate = Date()
        let entry = PaymentEntry(date: currentDate, payments: samplePayments)
        entries.append(entry)
        
        let timeline = Timeline(entries: entries, policy: .atEnd)
        completion(timeline)
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
}

// Widget entry
struct PaymentEntry: TimelineEntry {
    let date: Date
    let payments: [Payment]
}

// Payment row view
struct PaymentRow: View {
    let payment: Payment
    let isSmallWidget: Bool
    
    var body: some View {
        HStack(spacing: isSmallWidget ? 8 : 12) {
            // Date column with rounded square background
            VStack(alignment: .center, spacing: 0) {
                Text(payment.date.month)
                    .font(.system(size: isSmallWidget ? 10 : 12, weight: .medium))
                    .foregroundColor(.secondary)
                Text(payment.date.day)
                    .font(.system(size: isSmallWidget ? 14 : 16, weight: .semibold))
                    .foregroundColor(.secondary)
            }
            .frame(width: isSmallWidget ? 35 : 45, height: isSmallWidget ? 35 : 45)
            .background(Color(.systemGray6))
            .cornerRadius(isSmallWidget ? 8 : 10)
            
            // Main content and amount
            if isSmallWidget {
                // Simplified layout for small widget
                VStack(alignment: .leading, spacing: 2) {
                    Text(payment.merchant)
                        .font(.system(size: 14, weight: .semibold))
                    
                    HStack {
                        Text(payment.description)
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Text(payment.amount)
                            .font(.system(size: 14, weight: .semibold))
                    }
                }
            } else {
                // Standard layout for medium and large widgets
                VStack(alignment: .leading, spacing: 4) {
                    Text(payment.merchant)
                        .font(.system(size: 17, weight: .semibold))
                    Text(payment.description)
                        .font(.system(size: 15))
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(payment.amount)
                        .font(.system(size: 17, weight: .semibold))
                    Text(payment.paymentMethod)
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.vertical, isSmallWidget ? 8 : 12)
        .padding(.horizontal, isSmallWidget ? 8 : 12)
    }
}

// Widget view
struct PaymentWidgetView: View {
    var entry: Provider.Entry
    @Environment(\.widgetFamily) var widgetFamily
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Recent Payments")
                    .font(.system(size: widgetFamily == .systemSmall ? 14 : 16, weight: .bold))
                Spacer()
            }
            .padding(.horizontal, widgetFamily == .systemSmall ? 10 : 16)
            .padding(.top, widgetFamily == .systemSmall ? 10 : 16)
            .padding(.bottom, 8)
            
            // Payments
            VStack(spacing: 0) {
                ForEach(getPaymentsForSize()) { payment in
                    PaymentRow(payment: payment, isSmallWidget: widgetFamily == .systemSmall)
                    
                    if payment.id != getPaymentsForSize().last?.id {
                        Divider()
                            .padding(.horizontal, widgetFamily == .systemSmall ? 8 : 12)
                    }
                }
            }
            
            Spacer(minLength: 0)
        }
        .background(Color(.systemBackground))
    }
    
    // Return appropriate number of payments based on widget size
    private func getPaymentsForSize() -> [Payment] {
        switch widgetFamily {
        case .systemSmall:
            return Array(entry.payments.prefix(1))
        case .systemMedium:
            return Array(entry.payments.prefix(2))
        case .systemLarge:
            return Array(entry.payments.prefix(5))
        default:
            return Array(entry.payments.prefix(1))
        }
    }
}

// Widget configuration
struct PaymentWidget: Widget {
    let kind: String = "PaymentWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: Provider()) { entry in
            PaymentWidgetView(entry: entry)
        }
        .configurationDisplayName("Recent Payments")
        .description("View your most recent payments.")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge])
    }
}

// Preview providers
struct PaymentWidget_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            PaymentWidgetView(entry: PaymentEntry(date: Date(), payments: Provider().samplePayments))
                .previewContext(WidgetPreviewContext(family: .systemSmall))
                .previewDisplayName("Small")
            
            PaymentWidgetView(entry: PaymentEntry(date: Date(), payments: Provider().samplePayments))
                .previewContext(WidgetPreviewContext(family: .systemMedium))
                .previewDisplayName("Medium")
            
            PaymentWidgetView(entry: PaymentEntry(date: Date(), payments: Provider().samplePayments))
                .previewContext(WidgetPreviewContext(family: .systemLarge))
                .previewDisplayName("Large")
        }
    }
}
