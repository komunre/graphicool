pub fn u8vec_to_f32vec(input: Vec<u8>) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len() / 4);
    let mut i = 0;
    while i < input.len() {
        let arr = [input[i], input[i + 1], input[i + 2], input[i + 3]];
        let v = f32::from_le_bytes(arr);
        i += 4;
        output.push(v);
    }
    return output;
}

pub fn u8vec_to_u32vec(input: Vec<u8>) -> Vec<u32> {
    let mut output = Vec::with_capacity(input.len() / 4);
    let mut i = 0;
    while i < input.len() {
        let arr = [input[i], input[i + 1], input[i + 2], input[i + 3]];
        let v = u32::from_le_bytes(arr);
        i += 4;
        output.push(v);
    }
    return output;
}

pub fn u8vec_to_u16vec(input: Vec<u8>) -> Vec<u16> {
    let mut output = Vec::with_capacity(input.len() / 2);
    let mut i = 0;
    while i < input.len() {
        let arr = [input[i], input[i + 1]];
        let v = u16::from_le_bytes(arr);
        i += 2;
        output.push(v);
    }
    return output;
}
