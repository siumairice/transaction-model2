import SwiftUI
import WidgetKit

// Widget configuration
struct PaymentWidget: Widget {
    let kind: String = "PaymentWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: PaymentProvider()) { entry in
            PaymentWidgetView(entry: entry)
        }
        .configurationDisplayName("Recent Payments")
        .description("View your most recent transactions")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge])
    }
} 