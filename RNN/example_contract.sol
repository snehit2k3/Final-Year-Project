// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ReentrancySecure
 * @dev This contract is secured against re-entrancy attacks.
 * It follows the Checks-Effects-Interactions pattern.
 */
contract ReentrancySecure {

    mapping(address => uint) public userBalances;

    // Function to deposit Ether into the contract
    function deposit() public payable {
        userBalances[msg.sender] += msg.value;
    }

    // Function to check the balance of this contract
    function getContractBalance() public view returns (uint) {
        return address(this).balance;
    }

    /**
     * @dev This function is SECURE against re-entrancy.
     * It updates the user's balance to 0 BEFORE sending the Ether.
     */
    function withdraw(uint _amount) public {
        // 1. CHECKS: Validate all conditions first.
        require(userBalances[msg.sender] >= _amount, "Insufficient balance");

        // 2. EFFECTS: Update the state of the contract.
        // We set the balance to its new value immediately.
        userBalances[msg.sender] -= _amount;

        // If an attacker tries to re-enter now, their balance is already reduced,
        // so the `require` check at the top will fail.

        // 3. INTERACTIONS: Interact with external contracts last.
        (bool sent, ) = msg.sender.call{value: _amount}("");
        require(sent, "Failed to send Ether");
    }
}