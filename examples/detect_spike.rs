use chat_spike::{ChatSpikeDetector, Event};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Debug)]
pub struct Chat {
    pub msg: String,
    pub ts: DateTime<Utc>,
}

fn main() {
    let file =
        std::fs::File::open("examples/data/sample.json").expect("file should open read only");
    let chats: Vec<Chat> = serde_json::from_reader(file).expect("file should be proper JSON");
    let mut bursts = vec![];
    let mut ss = ChatSpikeDetector::<30, 100>::default().with_threshold(2.0, 1.0);
    let mut wall0: Option<DateTime<Utc>> = None;
    let instant0 = Instant::now();
    for (i, chat) in chats.iter().enumerate() {
        let instant = if let Some(wall0) = wall0 {
            instant0 + Duration::from_secs_f64((chat.ts - wall0).as_seconds_f64())
        } else {
            wall0 = Some(chat.ts);
            instant0
        };
        match ss.update_and_detect(chat.msg.clone(), instant) {
            Event::SpikeBegin {
                summary, surprise, ..
            } => {
                //println!("---- spike begin!");
                bursts.push((i, chat.ts, summary.unwrap().to_owned(), surprise, true))
            }
            Event::SpikeEnd {
                summary, surprise, ..
            } => {
                //println!("---- spike end!");
                bursts.push((i, chat.ts, summary.unwrap().to_owned(), surprise, false))
            }
            _ => (),
        };
        //println!("{:.2}~{}:{}", ss.current_surprise(), chat.ts, chat.msg);
    }
    println!(
        "detect {} spikes among {} chats, {} ~ {}",
        bursts.len() / 2,
        chats.len(),
        chats[0].ts,
        chats[chats.len() - 1].ts,
    );
    //println!("{:?}", bursts);
}
