import WidgetKit

// Provider for payment widget data
struct PaymentProvider: TimelineProvider {
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
} 