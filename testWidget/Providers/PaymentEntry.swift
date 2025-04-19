import WidgetKit

// Widget entry for payments
struct PaymentEntry: TimelineEntry {
    let date: Date
    let payments: [Payment]
} 