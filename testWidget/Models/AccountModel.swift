import SwiftUI
import Combine

// MARK: - Account Model
class AccountModel: ObservableObject, Identifiable {
    let id = UUID()
    
    @Published var accountName: String?
    @Published var accountNumber: String
    @Published var accessibleAccountNumber: String?
    @Published var balance: String
    
    init(accountName: String?, accountNumber: String, balance: String){
        self.accountName = accountName
        self.accountNumber = accountNumber
        self.accessibleAccountNumber = ""
        self.balance = balance
    }
    
    var displayName: String {
        accountName ?? "Account \(accountNumber)"
    }
    
    // Convert to AccountDetail
    func toAccountDetail() -> AccountDetail {
        return AccountDetail(
            id: self.id.uuidString,
            accountName: self.displayName,
            accountNumber: self.accountNumber,
            balance: self.balance
        )
    }
} 