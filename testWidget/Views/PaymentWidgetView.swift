import SwiftUI
import WidgetKit

// Widget view
struct PaymentWidgetView: View {
    var entry: PaymentEntry
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