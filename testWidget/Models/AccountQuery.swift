import AppIntents

struct AccountQuery: EntityQuery {
    func entities(for identifiers: [AccountDetail.ID]) async throws -> [AccountDetail] {
        // Get accounts from the shared store or fallback to static data
        return AccountDetail.getAllAccounts().filter { identifiers.contains($0.id) }
    }
} 