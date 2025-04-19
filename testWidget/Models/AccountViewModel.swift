import SwiftUI

// MARK: - Account View Model
class AccountViewModel {
    let account: AccountDetail
    let isHidden: Bool
    let monthlyChange: Double
    let month: String
    let currency: String
    
    init(account: AccountDetail, isHidden: Bool, monthlyChange: Double = 1200, month: String = "March", currency: String = "USD") {
        self.account = account
        self.isHidden = isHidden
        self.monthlyChange = monthlyChange
        self.month = month
        self.currency = currency
    }
    
    var accountType: String {
        account.accountName.contains("Investment") ? "Investment" : "Chequing"
    }
    
    var displayBalance: String {
        account.displayBalance(isHidden: isHidden)
    }
    
    var displayMonthlyChange: String {
        if isHidden {
            return "\(monthlyChange >= 0 ? "+" : "-")$****"
        } else {
            let prefix = monthlyChange >= 0 ? "+" : ""
            return "\(prefix)$\(String(format: "%.0f", monthlyChange))"
        }
    }
    
    var monthlyChangeColor: Color {
        monthlyChange >= 0 ? Color.green : Color.red
    }
} 