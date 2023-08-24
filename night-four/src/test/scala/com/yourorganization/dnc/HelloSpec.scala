package example

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfter
import org.scalatest.BeforeAndAfterAll
import com.yourorganization.dnc._

// Tests

class MemoryTest extends AnyFunSuite {

  test("read returns correct read vectors") {

    val memory = Tensor(
      Array(0.1f, 0.2f, 0.3f, 0.4f), // Known memory values
      Array(4, 1) 
    )
    
    val readWeights = Tensor(
      Array(0.5f, 0.5f), // Weights to extract first two values
      Array(1, 2)
    )

    val memoryModule = new Memory(2, 1)
    
    val output = memoryModule.read(readWeights)

    // Validate contents
    assert(output == Tensor(Array(0.1f, 0.2f), Array(1, 2))) 
 
  }

  test("write updates memory correctly") {

    // Known initial values
    val prevMemory = Tensor(Array(
      0.1f, 0.2f, 0.3f, 0.4f  
    ), Array(4, 1))  

    val writeWeights = Tensor(Array(1f, 0f, 0f, 1f), Array(1, 4)) 
    val eraseVec = Tensor(Array(0f, 0f, 0f, 0f), Array(1, 4))
    val addVec = Tensor(Array(1f, 1f, 1f, 1f), Array(1, 4))

    val memoryModule = new Memory(4, 1)
    
    val updatedMemory = memoryModule.write(
      writeWeights, eraseVec, addVec)

    // Validate computation      
    val expectedMemory = Tensor(Array(
      0.1f, 1f, 1f, 0.4f
    ), Array(4, 1))

    assert(updatedMemory == expectedMemory)

  }
  
}