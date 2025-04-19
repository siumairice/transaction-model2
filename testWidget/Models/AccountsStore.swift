import SwiftUI
import WidgetKit

// MARK: - Accounts Store
class AccountsStore: ObservableObject {
    @Published var accounts: [AccountDetail] = []
    static let shared = AccountsStore()
    
    private init() {
        // Try to load accounts from UserDefaults first
        if let loadedAccounts = loadAccountsFromUserDefaults(), !loadedAccounts.isEmpty {
            self.accounts = loadedAccounts
        } else {
            // Initialize with static data as fallback
            loadInitialAccounts()
        }
    }
    
    // Load initial accounts from static data
    func loadInitialAccounts() {
        self.accounts = AccountDetail.allAccounts
        
        // Save to UserDefaults for widget access
        saveAccountsToUserDefaults()
    }
    
    // Method to fetch accounts from API
    func fetchAccounts() async {
        // Simulate network delay for demo
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        
        // In a real app, you would make API calls here
        // For now, we'll just update with our static data
        await MainActor.run {
            self.accounts = AccountDetail.allAccounts
            
            // Save to UserDefaults for widget access
            saveAccountsToUserDefaults()
            
            // Reload widget timelines to reflect new data
            WidgetCenter.shared.reloadTimelines(ofKind: "AccountPreviewWidget")
            WidgetCenter.shared.reloadTimelines(ofKind: "BankingWidget")
        }
    }
    
    // Get all accounts - to be used by widget providers
    func getAllAccounts() -> [AccountDetail] {
        if accounts.isEmpty {
            // Try to load from UserDefaults
            if let loadedAccounts = loadAccountsFromUserDefaults(), !loadedAccounts.isEmpty {
                self.accounts = loadedAccounts
            } else {
                // Last resort fallback to static data
                self.accounts = AccountDetail.allAccounts
            }
        }
        return accounts
    }
    
    // Convert AccountModel to AccountDetail
    func convertToAccountDetail(from model: AccountModel) -> AccountDetail {
        return AccountDetail(
            id: model.id.uuidString,
            accountName: model.displayName,
            accountNumber: model.accountNumber,
            balance: model.balance
        )
    }
    
    // Save accounts to UserDefaults for widget access
    func saveAccountsToUserDefaults() {
        guard let userDefaults = UserDefaults(suiteName: BankWidgetConstants.appGroupSuite) else {
            return
        }
        
        // In a real app, you would encode your accounts array to Data
        // Here's a simple representation for demo purposes
        let accountsData: [[String: String]] = self.accounts.map { account in
            return [
                "id": account.id,
                "accountName": account.accountName,
                "accountNumber": account.accountNumber,
                "balance": account.balance
            ]
        }
        
        // Save as property list
        userDefaults.set(accountsData, forKey: BankWidgetConstants.Keys.accountsData)
        userDefaults.synchronize()
    }
    
    // Load accounts from UserDefaults
    func loadAccountsFromUserDefaults() -> [AccountDetail]? {
        guard let userDefaults = UserDefaults(suiteName: BankWidgetConstants.appGroupSuite),
              let accountsData = userDefaults.array(forKey: BankWidgetConstants.Keys.accountsData) as? [[String: String]] else {
            return nil
        }
        
        // Convert back to AccountDetail objects
        let accounts = accountsData.compactMap { accountDict -> AccountDetail? in
            guard let id = accountDict["id"],
                  let accountName = accountDict["accountName"],
                  let accountNumber = accountDict["accountNumber"],
                  let balance = accountDict["balance"] else {
                return nil
            }
            
            return AccountDetail(
                id: id,
                accountName: accountName,
                accountNumber: accountNumber,
                balance: balance
            )
        }
        
        return accounts.isEmpty ? nil : accounts
    }
} 