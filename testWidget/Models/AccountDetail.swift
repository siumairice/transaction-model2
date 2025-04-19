import SwiftUI
import AppIntents

// MARK: - AppEntity Model
struct AccountDetail: AppEntity {
    let id: String
    let accountName: String
    let accountNumber: String
    let balance: String
    let isAvailable = true
    
    static var typeDisplayRepresentation: TypeDisplayRepresentation = "Account"
    static var defaultQuery = AccountQuery()
            
    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(accountName) (\(accountNumber))")
    }
    
    // Display formatted balance based on visibility setting
    func displayBalance(isHidden: Bool) -> String {
        if isHidden {
            return "$****.**"
        } else {
            return balance
        }
    }

    // Mock data
    static let allAccounts: [AccountDetail] = [
        AccountDetail(id: "Checking", accountName: "My Checking Account", accountNumber: "1234", balance: "$5,432.10"),
        AccountDetail(id: "Savings", accountName: "Savings Account", accountNumber: "5678", balance: "$12,345.67"),
        AccountDetail(id: "Investment", accountName: "Investment Portfolio", accountNumber: "9012", balance: "$87,654.32"),
        AccountDetail(id: "Emergency", accountName: "Emergency Fund", accountNumber: "2468", balance: "$7,500.00"),
        AccountDetail(id: "Vacation", accountName: "Vacation Savings", accountNumber: "1357", balance: "$2,750.88"),
        AccountDetail(id: "Account9999", accountName: "Account 9999", accountNumber: "9999", balance: "$1,000.00")
    ]
    
    static func getAllAccounts() -> [AccountDetail] {
        // First try to get accounts from the store, if available
        if !AccountsStore.shared.accounts.isEmpty {
            return AccountsStore.shared.accounts
        }
        // Fallback to static data
        return self.allAccounts
    }
    
    // Converts to AccountModel
    func toAccountModel() -> AccountModel {
        return AccountModel(
            accountName: self.accountName,
            accountNumber: self.accountNumber,
            balance: self.balance
        )
    }
} 