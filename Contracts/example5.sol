// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title VulnerableBank
 * @dev This contract is a simple Ether bank that is HIGHLY VULNERABLE
 * to a classic reentrancy attack.
 *
 * THE FLAW: In the `withdraw` function, the external call to send Ether
 * is made *before* the user's balance in the 'balances' mapping is
 * updated. An attacker can use a malicious fallback function to
 * call `withdraw` repeatedly before their balance is ever set to zero.
 */
contract VulnerableBank {

    // --- State Variables ---
    
    address public owner;
    mapping(address => uint) public balances;
    bool public isActive;
    uint public minDepositAmount;

    // --- Events ---

    event Deposited(address indexed user, uint amount);
    event Withdrawn(address indexed user, uint amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // --- Modifiers ---

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    modifier whenActive() {
        require(isActive, "Contract is not active");
        _;
    }

    // --- Functions ---

    /**
     * @dev Sets the initial owner and contract state.
     */
    constructor() {
        owner = msg.sender;
        isActive = true;
        minDepositAmount = 0.01 ether; // Set a minimum deposit
    }

    /**
     * @dev Toggles the contract's active state. Only owner can call.
     */
    function toggleActive() public onlyOwner {
        isActive = !isActive;
    }

    /**
     * @dev Allows a user to deposit Ether into the bank.
     */
    function deposit() public payable whenActive {
        require(msg.value >= minDepositAmount, "Deposit is below minimum amount");
        
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    /**
     * @dev The VULNERABLE withdraw function.
     * Allows a user to withdraw a specified amount of Ether.
     */
    function withdraw(uint _amount) public whenActive {
        
        // 1. CHECK: Verify the user has enough funds
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        
        // 2. INTERACTION (THE FLAW):
        // The contract sends Ether to the 'msg.sender' *before*
        // updating its internal state (the 'balances' mapping).
        // If 'msg.sender' is a malicious contract, its fallback()
        // or receive() function will be triggered *right here*.
        // That fallback function can call this 'withdraw' function
        // again, and since the balance hasn't been updated, the
        // 'require' check on line 91 will pass again.
        (bool sent, ) = msg.sender.call{value: _amount}("");
        require(sent, "Failed to send Ether");
        
        // 3. EFFECT (TOO LATE):
        // This line is only reached *after* the external call
        // (and any potential re-entrant calls) have completed.
        // The attacker has already drained the funds.
        balances[msg.sender] -= _amount;
        
        emit Withdrawn(msg.sender, _amount);
    }

    /**
     * @dev Allows the owner to transfer ownership of the contract.
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    /**
     * @dev Helper function to check the balance of any address.
     */
    function getBalance(address user) public view returns (uint) {
        return balances[user];
    }

    /**
     * @dev Helper function to check the contract's total Ether balance.
     */
    function getContractBalance() public view returns (uint) {
        return address(this).balance;
    }

    /**
     * @dev Fallback function to receive Ether directly (e.g., from a transfer).
     */
    receive() external payable {
        deposit();
    }
}