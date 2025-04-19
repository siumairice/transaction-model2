import WidgetKit

// MARK: - Timeline Provider
struct AccountPreviewWidgetProvider: AppIntentTimelineProvider {
    func placeholder(in context: Context) -> AccountPreviewEntry {
        // Use accounts from store if available
        let availableAccounts = AccountDetail.getAllAccounts()
        let defaultAccounts = availableAccounts.count >= 2 ? 
                              [availableAccounts[0], availableAccounts[1]] : 
                              [AccountDetail.allAccounts[0], AccountDetail.allAccounts[1]]
        
        return AccountPreviewEntry(
            date: Date(),
            accounts: defaultAccounts,
            isBalancesHidden: false
        )
    }
    
    func snapshot(for configuration: SelectAccountsIntent, in context: Context) async -> AccountPreviewEntry {
        // Get accounts from store if available
        let availableAccounts = AccountDetail.getAllAccounts()
        
        // If no accounts selected, use the first two as placeholders
        let accounts = configuration.accounts.isEmpty
            ? (availableAccounts.count >= 2 ? [availableAccounts[0], availableAccounts[1]] : [AccountDetail.allAccounts[0], AccountDetail.allAccounts[1]])
            : configuration.accounts
            
        let userDefaults = UserDefaults(suiteName: BankWidgetConstants.appGroupSuite)
        let isHidden = userDefaults?.bool(forKey: BankWidgetConstants.Keys.balancesHidden) ?? false
            
        return AccountPreviewEntry(
            date: Date(),
            accounts: accounts,
            isBalancesHidden: isHidden
        )
    }
    
    func timeline(for configuration: SelectAccountsIntent, in context: Context) async -> Timeline<AccountPreviewEntry> {
        let userDefaults = UserDefaults(suiteName: BankWidgetConstants.appGroupSuite)
        let isHidden = userDefaults?.bool(forKey: BankWidgetConstants.Keys.balancesHidden) ?? false
        
        // Try to get accounts from configuration, or fall back to store
        var accounts = configuration.accounts
        if accounts.isEmpty {
            // Try to get accounts from store
            let availableAccounts = AccountDetail.getAllAccounts()
            accounts = availableAccounts.prefix(5).map { $0 }
        }
        
        // Create the timeline entry
        let entry = AccountPreviewEntry(
            date: Date(),
            accounts: accounts,
            isBalancesHidden: isHidden
        )
        
        // Update every hour
        let nextUpdate = Calendar.current.date(byAdding: .hour, value: 1, to: Date())!
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        return timeline
    }
} 