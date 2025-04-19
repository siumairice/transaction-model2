import SwiftUI
import WidgetKit

// MARK: - Widget View
struct AccountPreviewWidgetView: View {
    var entry: AccountPreviewEntry
    @Environment(\.widgetFamily) var family
    
    var body: some View {
        switch family {
        case .systemSmall:
            smallWidgetView
        case .systemMedium:
            mediumWidgetView
        case .systemLarge:
            largeWidgetView
        case .systemExtraLarge:
            largeWidgetView
        @unknown default:
            mediumWidgetView
        }
    }
    
    // Small Widget View
    var smallWidgetView: some View {
        VStack(spacing: 0) {
            // Use reusable small header
            SmallWidgetHeader(isHidden: entry.isBalancesHidden)
            
            if entry.accounts.isEmpty {
                EmptyAccountsView()
            } else {
                let account = entry.accounts[0]
                VStack(spacing: 2) {
                    Text(account.accountName)
                        .font(.system(size: 12))
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                        .lineLimit(1)
                    
                    Text(account.displayBalance(isHidden: entry.isBalancesHidden))
                        .font(.system(size: 20))
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                        .padding(.top, 1)
                    
                    Text("Account \(account.accountNumber)")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                        .padding(.top, 2)
                }
                .padding(10)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
    
    // Medium Widget View
    var mediumWidgetView: some View {
        VStack(spacing: 0) {
            BankingWidgetHeader(isHidden: entry.isBalancesHidden)
            
            if entry.accounts.isEmpty {
                EmptyAccountsView()
            } else {
                let displayAccounts = Array(entry.accounts.prefix(2))
                VStack(spacing: 0) {
                    ForEach(displayAccounts, id: \.id) { account in
                        AccountRowView(account: account, isHidden: entry.isBalancesHidden)
                        
                        if account.id != displayAccounts.last?.id {
                            Divider()
                                .padding(.horizontal, 12)
                        }
                    }
                }
            }
        }
    }
    
    // Large Widget View
    var largeWidgetView: some View {
        VStack(spacing: 0) {
            BankingWidgetHeader(isHidden: entry.isBalancesHidden)
            
            if entry.accounts.isEmpty {
                EmptyAccountsView()
            } else {
                let displayAccounts = Array(entry.accounts.prefix(5))
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(displayAccounts, id: \.id) { account in
                            AccountRowView(account: account, isHidden: entry.isBalancesHidden)
                            
                            if account.id != displayAccounts.last?.id {
                                Divider()
                                    .padding(.horizontal, 12)
                            }
                        }
                    }
                }
            }
        }
    }
} 