import AppIntents

// MARK: - Configuration Intent
struct SelectAccountsIntent: WidgetConfigurationIntent {
    static var title: LocalizedStringResource = "Select Accounts"
    static var description = IntentDescription("Selects the bank accounts to display information for.")
    
    @Parameter(
            title: "Accounts",
            description: "Select accounts to display in the widget",
            optionsProvider: AccountOptionsProvider()
        )
    var accounts: [AccountDetail]
    
    init(accounts: [AccountDetail]) {
        self.accounts = accounts
    }

    init() {
        self.accounts = []
    }
} 