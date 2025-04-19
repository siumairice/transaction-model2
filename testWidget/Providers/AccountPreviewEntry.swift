import WidgetKit

// MARK: - Widget Entry
struct AccountPreviewEntry: TimelineEntry {
    let date: Date
    let accounts: [AccountDetail]
    let isBalancesHidden: Bool
} 