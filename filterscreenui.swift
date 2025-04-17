import SwiftUI

struct Transaction: Identifiable {
    let id = UUID()
    let date: String
    let description: String
    let amount: Double
    let isDebit: Bool
    let categories: [TransactionCategory]
}

// Group transactions by date
struct TransactionGroup: Identifiable {
    let id = UUID()
    let date: String
    let transactions: [Transaction]
}

enum TransactionCategory: String, CaseIterable, Identifiable {
    case all = "All"
    case grocery = "Grocery"
    case gym = "Gym"
    case subscription = "Subscription"
    case dining = "Dining"
    case entertainment = "Entertainment"
    case utilities = "Utilities"
    case pets = "Pets"
    case shopping = "Shopping"
    case electronics = "Electronics"
    case housing = "Housing"
    case transportation = "Transportation"
    
    var id: String { self.rawValue }
    
    var icon: String {
        switch self {
        case .all: return "list.bullet"
        case .grocery: return "cart"
        case .gym: return "dumbbell"
        case .subscription: return "arrow.up.arrow.down"
        case .dining: return "fork.knife"
        case .entertainment: return "tv"
        case .utilities: return "bolt"
        case .pets: return "pawprint"
        case .shopping: return "bag"
        case .electronics: return "ipad"
        case .housing: return "house"
        case .transportation: return "car"
        }
    }
}

struct SearchBarWithTagsTransactionsView: View {
    @State private var searchText = ""
    @State private var isSearching = false
    @State private var selectedCategories: Set<TransactionCategory> = []
    
    let rbcBlue = Color(red: 0/255, green: 106/255, blue: 195/255)
    
    let transactions = [
        Transaction(date: "Apr 17, 2025", description: "STAFF - PAYROLL", amount: 1602.61, isDebit: false, categories: [.all]),
        Transaction(date: "Apr 17, 2025", description: "UBER RIDE", amount: 18.75, isDebit: true, categories: [.transportation]),
        Transaction(date: "Apr 16, 2025", description: "STARBUCKS COFFEE", amount: 6.45, isDebit: true, categories: [.dining]),
        Transaction(date: "Apr 15, 2025", description: "WALMART GROCERY", amount: 127.38, isDebit: true, categories: [.grocery, .subscription]),
        Transaction(date: "Apr 15, 2025", description: "PETCO SUPPLIES", amount: 42.19, isDebit: true, categories: [.pets]),
        Transaction(date: "Apr 14, 2025", description: "SPOTIFY PREMIUM", amount: 9.99, isDebit: true, categories: [.subscription, .entertainment]),
        Transaction(date: "Apr 12, 2025", description: "FITNESS PLUS MONTHLY", amount: 49.99, isDebit: true, categories: [.gym, .subscription]),
        Transaction(date: "Apr 12, 2025", description: "TARGET SHOPPING", amount: 89.23, isDebit: true, categories: [.shopping]),
        Transaction(date: "Apr 10, 2025", description: "NETFLIX SUBSCRIPTION", amount: 15.99, isDebit: true, categories: [.subscription, .entertainment]),
        Transaction(date: "Apr 10, 2025", description: "CHEVRON GAS STATION", amount: 48.50, isDebit: true, categories: [.transportation]),
        Transaction(date: "Apr 8, 2025", description: "CHIPOTLE RESTAURANT", amount: 23.47, isDebit: true, categories: [.dining]),
        Transaction(date: "Apr 8, 2025", description: "APPLE STORE", amount: 129.00, isDebit: true, categories: [.shopping, .electronics]),
        Transaction(date: "Apr 5, 2025", description: "ELECTRICITY BILL", amount: 142.76, isDebit: true, categories: [.utilities]),
        Transaction(date: "Apr 5, 2025", description: "WATER BILL", amount: 67.89, isDebit: true, categories: [.utilities]),
        Transaction(date: "Apr 3, 2025", description: "STAFF - PAYROLL", amount: 1602.61, isDebit: false, categories: [.all]),
        Transaction(date: "Apr 3, 2025", description: "PHONE BILL", amount: 85.42, isDebit: true, categories: [.utilities]),
        Transaction(date: "Apr 2, 2025", description: "Online Transfer to Deposit Account-9541", amount: 2895.00, isDebit: true, categories: [.all]),
        Transaction(date: "Apr 1, 2025", description: "AMAZON PRIME", amount: 14.99, isDebit: true, categories: [.subscription, .entertainment]),
        Transaction(date: "Apr 1, 2025", description: "RENT PAYMENT", amount: 1450.00, isDebit: true, categories: [.housing])
    ]
    
    // Group transactions by date
    private var groupedTransactions: [TransactionGroup] {
        let grouped = Dictionary(grouping: filteredTransactions) { $0.date }
        return grouped.map { TransactionGroup(date: $0.key, transactions: $0.value) }
            .sorted { $0.date > $1.date } // Sort by date descending
    }
    
    var filteredTransactions: [Transaction] {
        if selectedCategories.isEmpty {
            return transactions.filter {
                searchText.isEmpty ? true : $0.description.lowercased().contains(searchText.lowercased())
            }
        } else {
            return transactions.filter { transaction in
                transaction.categories.contains { selectedCategories.contains($0) } &&
                (searchText.isEmpty || transaction.description.lowercased().contains(searchText.lowercased()))
            }
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Search Bar (unchanged)
            VStack(spacing: 0) {
                Divider()
                    .background(Color.gray.opacity(0.3))
                
                HStack {
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(Color.gray)
                            .padding(.leading, 8)
                        
                        TextField("Search", text: $searchText)
                            .padding(.vertical, 10)
                    }
                    .background(Color(UIColor.systemGray5))
                    .cornerRadius(10)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 16)
                    
                    if isSearching {
                        Button("Cancel") {
                            searchText = ""
                            isSearching = false
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                        .foregroundColor(rbcBlue)
                        .padding(.trailing, 16)
                        .transition(.move(edge: .trailing))
                        .animation(.default)
                    }
                }
                
                Divider()
                    .background(Color.gray.opacity(0.3))
            }
            .onTapGesture {
                isSearching = true
            }
            
            // Category Tags (unchanged)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    HStack(spacing: 4) {
                        Image(systemName: "slider.horizontal.3")
                            .foregroundColor(rbcBlue)
                        Text("Filter")
                            .font(.system(size: 17, weight: .medium))
                            .foregroundColor(rbcBlue)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)

                    ForEach(TransactionCategory.allCases.filter { $0 != .all }) { category in
                        CategoryTag(
                            category: category,
                            isSelected: selectedCategories.contains(category),
                            action: {
                                if selectedCategories.contains(category) {
                                    selectedCategories.remove(category)
                                } else {
                                    selectedCategories.insert(category)
                                }
                            },
                            tagColor: rbcBlue
                        )
                    }
                }
                .padding(.vertical, 10)
                .padding(.trailing, 12)
            }

            // Selected tags summary (unchanged)
            if !selectedCategories.isEmpty {
                HStack {
                    Text("Filters: ")
                        .font(.system(size: 14))
                        .foregroundColor(.secondary)
                    
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(Array(selectedCategories)) { category in
                                HStack(spacing: 4) {
                                    Text(category.rawValue)
                                        .font(.system(size: 14))
                                        .foregroundColor(rbcBlue)
                                    
                                    Button(action: {
                                        selectedCategories.remove(category)
                                    }) {
                                        Image(systemName: "xmark.circle.fill")
                                            .font(.system(size: 14))
                                            .foregroundColor(rbcBlue.opacity(0.8))
                                    }
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(rbcBlue.opacity(0.1))
                                .cornerRadius(12)
                            }
                            
                            Button(action: {
                                selectedCategories.removeAll()
                            }) {
                                Text("Clear All")
                                    .font(.system(size: 14))
                                    .foregroundColor(rbcBlue)
                            }
                            .padding(.leading, 4)
                        }
                    }
                    .padding(.leading, 4)
                    
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
            }
            
            Divider()
                .background(Color.gray.opacity(0.3))
            
            // Transactions List - Updated to use grouped transactions
            if groupedTransactions.isEmpty {
                VStack {
                    Spacer()
                    Text("No transactions found")
                        .foregroundColor(.secondary)
                    Spacer()
                }
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(groupedTransactions) { group in
                            // Date header
                            ZStack {
                                Rectangle()
                                    .fill(Color(UIColor.systemGray6))
                                    .frame(maxWidth: .infinity)
                                
                                HStack {
                                    Text(group.date)
                                        .font(.system(size: 13))
                                        .foregroundColor(.secondary)
                                        .padding(.vertical, 12)
                                        .padding(.horizontal, 16)
                                    
                                    Spacer()
                                }
                            }
                            
                            // Transactions for this date
                            ForEach(group.transactions) { transaction in
                                TransactionRowView(transaction: transaction, rbcBlue: rbcBlue)
                            }
                        }
                    }
                }
            }
        }
    }
}

struct CategoryTag: View {
    let category: TransactionCategory
    let isSelected: Bool
    let action: () -> Void
    let tagColor: Color
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: category.icon)
                Text(category.rawValue)
                    .font(.system(size: 16))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Color.blue.opacity(isSelected ? 0.3 : 0.05))
            .clipShape(RoundedRectangle(cornerRadius: 13, style: .continuous))
        }
    }
}

struct TransactionRowView: View {
    let transaction: Transaction
    let rbcBlue: Color
    
    var body: some View {
        VStack(spacing: 0) {
            
            // Transaction details
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading) {
                        Text(transaction.description)
                            .font(.system(size: 17))
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Text(transaction.isDebit ? "-\(formatAmount(transaction.amount))" : formatAmount(transaction.amount))
                        .font(.system(size: 17))
                        .foregroundColor(.secondary)
                }
                
                // Category tags for this transaction
                if !transaction.categories.contains(.all) {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 4) {
                            ForEach(transaction.categories, id: \.self) { category in
                                HStack(spacing: 4) {
                                    Image(systemName: category.icon)
                                        .font(.system(size: 10))
                                    Text(category.rawValue)
                                        .font(.system(size: 12))
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(rbcBlue.opacity(0.1))
                                .foregroundColor(rbcBlue)
                                .cornerRadius(10)
                            }
                        }
                    }
                }
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 16)
            .background(Color.white)
            
            Divider()
                .background(Color.gray.opacity(0.3))
        }
    }
    
    private func formatAmount(_ amount: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.minimumFractionDigits = 2
        formatter.maximumFractionDigits = 2
        
        if let formattedAmount = formatter.string(from: NSNumber(value: amount)) {
            return formattedAmount
        }
        
        return String(format: "%.2f", amount)
    }
}

struct ContentView: View {
    var body: some View {
        SearchBarWithTagsTransactionsView()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
