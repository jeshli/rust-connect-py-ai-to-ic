
use std::cell::RefCell;
use tract_onnx::FrameworkExtension;
use tract_onnx::prelude::tvec;
use tract_onnx::prelude::tract_ndarray;
use tract_onnx::prelude::IntoTensor;
use tract_onnx::prelude::InferenceModelExt;



type SimplePlanTypeRead = tract_core::model::graph::Graph<
    tract_hir::infer::fact::InferenceFact,
    std::boxed::Box<
        dyn tract_hir::infer::ops::InferenceOp
    >
>;

type SimplePlanTypeRun = tract_core::plan::SimplePlan<
    tract_core::model::fact::TypedFact,
    std::boxed::Box<dyn tract_core::ops::TypedOp>,
    tract_core::model::graph::Graph<
        tract_core::model::fact::TypedFact,
        std::boxed::Box<dyn tract_core::ops::TypedOp>
    >
>;

type Tensor = tract_data::tensor::Tensor;

thread_local! {
    static WASM_REF_CELL: RefCell<Vec<u8>> = RefCell::new(vec![]);
    static MODEL_READ_REF_CELL: RefCell<Option<SimplePlanTypeRead>> = RefCell::new(None);
    //static MODEL_PIPELINE: RefCell<Option<ModelPipeline>> = RefCell::new(None);
    static MODEL: RefCell<Option<SimplePlanTypeRun>> = RefCell::new(None);
    //static MODEL_PIPELINE: RefCell<Option<Vec<SimplePlanTypeRun>>> = RefCell::new(None);

}


/*
##################
Model Pipeline
###################
*/

struct ModelPipeline {
    models: Vec<SimplePlanTypeRun>,
}


impl ModelPipeline {
    fn add_model(&mut self, model: SimplePlanTypeRun) {
        self.models.push(model);
    }
    pub fn clear(&mut self) {
        self.models.clear();
    }
    pub fn new() -> Self {
        ModelPipeline {
            models: Vec::new(),
        }
    }

    pub fn run_model_at_index(&self, index: u8, input_tensor: Tensor) -> Result<(Vec<f32>, Vec<usize>), String> {

        let model = &self.models[index as usize];
        let result = model.run(tvec!(input_tensor.into()))
            .map_err(|e| e.to_string())?; // Handle the error properly

        if let Some(first_tensor) = result.first() {
            let shape = first_tensor.shape().to_vec(); // Get the shape of the tensor
            match first_tensor.to_array_view::<f32>() {
                Ok(values) => Ok((values.as_slice().unwrap_or(&[]).to_vec(), shape)),
                Err(_) => Err("Failed to convert tensor to array view".to_string()),
            }
        } else {
            Err("No Data in Result".to_string())
        }
    }

}

/*
#[ic_cdk::update]
fn initialize_model_pipeline() {
    MODEL_PIPELINE.with(|pipeline_ref| {
        let mut pipeline = pipeline_ref.borrow_mut();
        *pipeline = Some(ModelPipeline::new()); // Assuming ModelPipeline has a new() method
    });
}


fn add_model_to_pipeline(model: SimplePlanTypeRun) {
    MODEL_PIPELINE.with(|pipeline_ref| {
        let mut pipeline = pipeline_ref.borrow_mut();
        if let Some(model_pipeline) = &mut *pipeline {
            model_pipeline.add_model(model);
        }
    });
}
*/


/*
#############################
Uploading and Initializing Model
#############################
*/

#[ic_cdk::update]
pub fn upload_model_chunks(bytes: Vec<u8>) { // -> Result<(), String>
    WASM_REF_CELL.with(|wasm_ref_cell| {
        let mut wasm_ref_mut = wasm_ref_cell.borrow_mut();
        wasm_ref_mut.extend(bytes);
    });
}

#[ic_cdk::update]
fn model_bytes_to_plan() {
    let model_bytes = match call_model_bytes() {
        Ok(value) => value,
        Err(_) => bytes::Bytes::new(),  // Return empty bytes in case of error
    };
    let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes).expect("Failed to Read");
    MODEL_READ_REF_CELL.with(|model_ref_cell| {
        *model_ref_cell.borrow_mut() = Some(bytes_model);
    });
}

fn call_model_bytes() -> Result<bytes::Bytes, String> {
    WASM_REF_CELL.with(|wasm_ref_cell| {
        // Use `std::mem::take` to replace the contents of the cell with an empty vector
        // and return the original contents.
        let wasm_ref = std::mem::take(&mut *wasm_ref_cell.borrow_mut());

        // Convert the vector into a `Bytes` instance. This avoids cloning as the vector's
        // ownership is transferred to the `Bytes` instance.
        let model_bytes = bytes::Bytes::from(wasm_ref);

        Ok(model_bytes)
    })
}


#[ic_cdk::update]
fn plan_to_running_model() {

    MODEL_READ_REF_CELL.with(|read_cell| {
        // Take the model out of the RefCell, leaving None in its place
        if let Some(read_model) = read_cell.borrow_mut().take() {
            let run_model = read_model.into_optimized()
                .expect("Couldn't Optimize")
                .into_runnable()
                .expect("Couldn't Make Runnable");

            set_model(run_model);



        }
    })

}

fn set_model(model: SimplePlanTypeRun) {
    MODEL.with(|model_ref_cell| {
        *model_ref_cell.borrow_mut() = Some(model);
    });
}


/*
#############################
Canister Queries
#############################
*/



#[ic_cdk::query]
async fn word_embeddings(input_text: String) -> Vec<f32> {

    // Trim, remove brackets, split by comma, and parse each number
    let numbers_result: Result<Vec<i64>, _> = input_text
        .trim()                        // Trim whitespace
        .trim_matches(|c| c == '[' || c == ']')  // Remove brackets
        .split(',')                    // Split by comma
        .map(|num_str| num_str.trim().parse::<i64>()) // Parse each number
        .collect();                    // Collect into a Result<Vec<i64>, E>

    // Handle the result
    let numbers = match numbers_result {
        Ok(vec) => vec,
        Err(e) => {
            // Handle the error
            //eprintln!("Failed to parse number: {}", e);
            ic_cdk::println!("Failed to parse number: {}", e);
            vec![]  // Return an empty vector or handle as needed
        },
    };

    let output: Vec<f32> = match run_model_and_get_result_chain(numbers) {
    Ok(result) => result,
    Err(e) => {
        ic_cdk::println!("Failed: {}", e);
        vec![-1.0] // Return a default vector as specified
    }
    };

    output

}


fn run_model_and_get_result_chain(token_ids: Vec<i64>) -> Result<Vec<f32>, String> {
    let input_shape = vec![1, token_ids.len()];

    let input_tensor = match tract_ndarray::Array::from_shape_vec(input_shape, token_ids) {
        Ok(array) => array.into_tensor(),
        Err(_) => return Err("Failed to create tensor from shape and values".into()),
    };

    // Use a match statement to handle the Result from run_model
    //match run_model(0_u8, input_tensor) {
    match run_model(input_tensor) {
        Ok((out, _result_shape)) => Ok(out),
        Err(e) => {
            // Print the error and possibly return or handle it
            println!("Error encountered: {}", e);
            // You can decide to return the error or handle it differently
            Err(e)
        }
    }

}

/*
fn run_model(index: u8, input: Tensor) -> Result<(Vec<f32>, Vec<usize>), String> {
    MODEL_PIPELINE.with(|pipeline_ref| {
        let pipeline = pipeline_ref.borrow();
        if let Some(model_pipeline) = &*pipeline {
            match model_pipeline.run_model_at_index(index, input) {
                Ok(result_tensor) => Ok(result_tensor), // Return Ok variant wrapping the successful result
                Err(e) => Err(format!("Model computation failed: {:?}", e)), // Return an Err variant with an error message
            }
        } else {
            // Return an Err variant if the model pipeline is not initialized
            Err("Model pipeline is not initialized.".to_string())
        }
    })
}
*/




fn run_model(input: Tensor) -> Result<(Vec<f32>, Vec<usize>), String> {
    MODEL.with(|model_ref| {
        if let Some(model) = &*model_ref.borrow() {
            model.run(tvec!(input.into()))
                .map_err(|e| e.to_string())
                .and_then(|result| {
                    result.first().map_or(
                        Err("No Data in Result".to_string()),
                        |first_tensor| {
                            let shape = first_tensor.shape().to_vec();
                            // Here, ensure we're correctly constructing the tuple as expected
                            first_tensor.to_array_view::<f32>()
                                .map(|values| (values.as_slice().unwrap_or(&[]).to_vec(), shape))
                                // No need to map to values alone, return the tuple as is
                                .map_err(|_| "Failed to convert tensor to array view".to_string())
                        }
                    )
                })
        } else {
            Err("Model is not initialized.".to_string())
        }
    })
}


/*
        let result = model.run(tvec!(input_tensor.into()))
            .map_err(|e| e.to_string())?; // Handle the error properly

        if let Some(first_tensor) = result.first() {
            let shape = first_tensor.shape().to_vec(); // Get the shape of the tensor
            match first_tensor.to_array_view::<f32>() {
                Ok(values) => Ok((values.as_slice().unwrap_or(&[]).to_vec(), shape)),
                Err(_) => Err("Failed to convert tensor to array view".to_string()),
            }
        } else {
            Err("No Data in Result".to_string())
        }
*/