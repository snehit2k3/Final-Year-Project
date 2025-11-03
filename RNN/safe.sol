// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SecureSavingsPot
 * @dev This contract is a secured version of the VulnerableSavingsPot.
 * It uses the "Checks-Effects-Interactions" pattern and a reentrancy guard
 * to prevent reentrancy attacks.
 */
contract SecureSavingsPot {
    // Mapping of user addresses to their balances
    mapping(address => uint256) public userBalances;

    // Mapping to track deposit timestamps for a pseudo-reward system
    mapping(address => uint256) public depositTimestamps;

    // A small reward rate for holding funds
    uint256 public constant REWARD_RATE_PER_SECOND = 0.00001 ether;

    // Reentrancy guard state variable
    bool private locked;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    /**
     * @dev A modifier to prevent reentrancy attacks.
     */
    modifier nonReentrant() {
        require(!locked, "No re-entrancy");
        locked = true;
        _;
        locked = false;
    }

    /**
     * @dev Allows a user to deposit Ether into the contract.
     */
    function deposit() external payable {
        require(msg.value > 0, "Deposit amount must be greater than zero.");
        userBalances[msg.sender] += msg.value;
        if (depositTimestamps[msg.sender] == 0) {
            depositTimestamps[msg.sender] = block.timestamp;
        }
        emit Deposited(msg.sender, msg.value);
    }

    /**
     * @dev Calculates a simple time-based reward.
     */
    function calculateReward(address user) public view returns (uint256) {
        if (depositTimestamps[user] == 0) {
            return 0;
        }
        uint256 timeHeld = block.timestamp - depositTimestamps[user];
        return timeHeld * REWARD_RATE_PER_SECOND;
    }

    /**
     * @dev SECURE WITHDRAW FUNCTION
     * Implements the Checks-Effects-Interactions pattern and uses a reentrancy guard.
     */
    function withdraw() external nonReentrant {
        // --- 1. Checks ---
        uint256 balance = userBalances[msg.sender];
        require(balance > 0, "You have no funds to withdraw.");

        uint256 reward = calculateReward(msg.sender);
        uint256 totalWithdrawAmount = balance + reward;

        // --- 2. Effects (State changes are made BEFORE the external call) ---
        userBalances[msg.sender] = 0;
        depositTimestamps[msg.sender] = 0; // Reset timestamp

        // If the attacker calls back into this function, the balance is already 0,
        // so the initial `require` check will fail. The `nonReentrant` modifier
        // also prevents the call from entering again.

        emit Withdrawn(msg.sender, totalWithdrawAmount);

        // --- 3. Interaction (External call is the last step) ---
        (bool success, ) = msg.sender.call{value: totalWithdrawAmount}("");
        require(success, "Failed to send Ether.");
    }

    /**
     * @dev A helper function to check the balance of this contract.
     */
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}