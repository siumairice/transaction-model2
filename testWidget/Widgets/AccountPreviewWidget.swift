import SwiftUI
import WidgetKit

// MARK: - Widget Configuration
struct AccountPreviewWidget: Widget {
    let kind: String = "AccountPreviewWidget"
    @Environment(\.widgetFamily) var family
    
    var body: some WidgetConfiguration {
        let config = AppIntentConfiguration(
            kind: kind,
            intent: SelectAccountsIntent.self,
            provider: AccountPreviewWidgetProvider()) { entry in
            AccountPreviewWidgetView(entry: entry)
        }
        .configurationDisplayName("Banking Accounts")
        .description(getConfigurationDescription())
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge, .systemExtraLarge])
        
        // Apply contentMarginsDisabled only on iOS 17+
        if #available(iOS 17.0, *) {
            return config.contentMarginsDisabled()
        } else {
            return config
        }
    }
    
    func getConfigurationDescription() -> String {
        switch family {
        case .systemSmall:
            "Select an account to display"
        case .systemMedium:
            "Select 2 accounts to display"
        case .systemLarge:
            "Select up to 5 accounts to display"
        case .systemExtraLarge:
            "Select up to 5 accounts to display"
        @unknown default:
            "Select an account to display"
        }
    }
} 