import AppIntents

// MARK: - Dynamic Options Provider for Accounts
struct AccountOptionsProvider: DynamicOptionsProvider {
    func results() async throws -> [AccountDetail] {
        // In a real app, you might fetch these from a database or API
        // Get accounts from the shared store
        return AccountDetail.getAllAccounts()
    }
} 