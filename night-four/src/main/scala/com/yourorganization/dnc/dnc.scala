package com.yourorganization.dnc

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.ann._
import org.apache.spark.ml.PipelineStage
import scala.collection.mutable.ListBuffer

// Controller
class Controller(inputDim: Int, hiddenSize: Int) extends Serializable {

  val lstm = LSTM(hiddenSize) // LSTM layer
  
  def forward(input: Tensor, state: LSTMState): (Tensor, LSTMState) = {
    // Pass input through LSTM 
    val (output, newState) = lstm.forward(input, state)  
    (output, newState)
  }
}

// Memory Class

class Memory(numSlots: Int, wordSize: Int) extends Serializable {

  val memory = Tensor.zeros(numSlots, wordSize)

  def read(readWeights: Tensor): Tensor = {

    // Use readWeights consistently    
    require(readWeights.dim == 3, "Read weights should have 3 dims")
    
    // Split and accumulate read vectors

    var readVecs = ListBuffer[Tensor]()

    for (headWeights <- readWeights.split(1)) {
      // Calculate each read vector  
      val headReadVec = headWeights.transpose(1, 2).matmul(memory)
      readVecs += headReadVec 
    }

    // Stack into one tensor
    Tensor.stack(readVecs.toSeq, 1) 
  }

  def write(writeWeights: Tensor, eraseVec: Tensor, addVec: Tensor): Tensor = {
    
    // Don't mutate existing memory 
    // Return a new memory tensor
    val expandedWriteWeights = writeWeights.expand(Array(-1, numSlots))
    val expandedEraseVec = eraseVec.expand(Array(-1, 1, wordSize))
    val retention = 1 - (expandedWriteWeights * expandedEraseVec)
    val updatedMemory = (memory * retention) + 
      (expandedWriteWeights.matmul(addVec.expand(Array(-1, 1, wordSize))))
      
    updatedMemory
  }

}

// Temporal linkage matrix
class TemporalLinkageMatrix(numSlots: Int) extends Serializable {

  // Matrix to store temporal linkages
  val linkage = Tensor.zeros(numSlots, numSlots)

  def update(prevWriteWeights: Tensor, writeWeights: Tensor): Tensor = {
    // Implement update to linkage matrix
    // Based on equations in dataflow plan 
  }

}

// DNCTwo
class DNCTwo(
  inputDim: Int,
  numSlots: Int,
  wordSize: Int,
  numReads: Int,
  hiddenSize: Int
) extends Serializable {

  // Components
  val controller = new Controller(inputDim, hiddenSize)
  val memory = new Memory(numSlots, wordSize)
  val linkage = new TemporalLinkageMatrix(numSlots)
  
  // Read/write params
  val readKeys = Linear(hiddenSize, numReads * wordSize)
  val readStrengths = Linear(hiddenSize, numReads)
  // ... other params
  
  def forward(input: Tensor, state: DNCState): (Tensor, DNCState) = {
    // 1. Pass through controller
    val controllerOut = controller.forward(input, state.controllerState)
    
    // 2. Get read/write params
    val readParams = getReadParams(controllerOut) 
    val writeParams = getWriteParams(controllerOut)
    
    // 3. Read from memory 
    val readVectors = memory.read(readParams) 
    
    // 4. Write to memory
    val updatedMemory = memory.write(writeParams)
    
    // 5. Update linkage matrix
    val updatedLinkage = linkage.update(state.writeWeights, writeParams)
    
    // 6. Update state
    val newState = DNCState(
      updatedMemory, readVectors, readParams, 
      writeParams, updatedLinkage, state.precedenceWeights,  
      controller.getState()
    )
    
    // 7. Prepare output
    val output = prepareOutput(controllerOut, readVectors)
    
    (output, newState)
  }
  
}

// DNC state
case class DNCState(
  memory: Tensor,
  readVectors: Tensor,
  readWeights: Tensor,
  writeWeights: Tensor,
  linkage: Tensor,
  precedenceWeights: Tensor,
  controllerState: LSTMState
)