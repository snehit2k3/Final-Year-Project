// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CrossFunction {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // This function is vulnerable, just like the classic example
    function withdraw(uint _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");

        // FLAW: Interaction before Effect
        (bool sent, ) = msg.sender.call{value: _amount}("");
        require(sent, "Failed to send Ether");

        balances[msg.sender] -= _amount;
    }

    // Attacker's fallback() can call this function while withdraw() is still running
    function transfer(address to, uint amount) public {
        // The check passes because balances[msg.sender] hasn't been reduced yet
        require(balances[msg.sender] >= amount, "Insufficient funds");
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}