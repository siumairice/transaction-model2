import AppIntents
import WidgetKit

// MARK: - Toggle Visibility Intent
struct ToggleBalanceVisibilityIntent: AppIntent {
    static var title: LocalizedStringResource = "Toggle Balance Visibility"
    
    func perform() async throws -> some IntentResult {
        let userDefaults = UserDefaults(suiteName: BankWidgetConstants.appGroupSuite)
        let currentlyHidden = userDefaults?.bool(forKey: BankWidgetConstants.Keys.balancesHidden) ?? false
        userDefaults?.set(!currentlyHidden, forKey: BankWidgetConstants.Keys.balancesHidden)
        userDefaults?.synchronize()
        return .result()
    }
}

// MARK: - Refresh Widget Intent
struct RefreshWidgetIntent: AppIntent {
    static var title: LocalizedStringResource = "Refresh Widget"
    
    func perform() async throws -> some IntentResult {
        // Fetch fresh accounts data from API
        await AccountsStore.shared.fetchAccounts()
        
        // Save to UserDefaults for widget access
        AccountsStore.shared.saveAccountsToUserDefaults()
        
        // Reload widget timelines
        WidgetCenter.shared.reloadTimelines(ofKind: "AccountPreviewWidget")
        return .result()
    }
} 