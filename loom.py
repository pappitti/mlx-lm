import mlx.core as mx
from mlx_lm.utils import load
import json
import argparse
import logging

DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-6bit"

# Simplified logs
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class Loom:
    def __init__(self, model, tokenizer, initial_prompt_text, depth, width):
        self.model = model
        self.tokenizer = tokenizer
        self.depth = depth
        self.width = width
        
        initial_ids = tokenizer.encode(initial_prompt_text)
        
        self.nb_layers = 0
        first_layer = {
            "layer": self.nb_layers,
            "id": f"{self.nb_layers}_0",
            "parent_id": None,
            "token_ids": initial_ids,
            "text": initial_prompt_text,
            "probability": 1.0,
            "last_token_text": "",
            "status": "pending",
            "children": [],
        }
        
        self.layer_map = {self.nb_layers: {first_layer["id"]: first_layer}}
        self.pending_items = [first_layer["id"]]
        
        # Temporary storage for items currently being processed in the batch
        self.processing_queue = []

    def _get_layer_index(self, item_id):
        return int(item_id.split("_")[0])

    def _delete_children(self, item_id):
        """
        Recursively deletes all descendants of item_id.
        Removes them from layer_map and pending_items.
        """
        layer_idx = self._get_layer_index(item_id)
        # Check if item exists (it might have been deleted by a parent call already)
        if item_id not in self.layer_map.get(layer_idx, {}):
            return

        item = self.layer_map[layer_idx][item_id]
        
        # Iterate over a copy of the children list
        children_to_delete = list(item["children"])
        
        for child_id in children_to_delete:
            # 1. Recursively delete grandchildren first
            self._delete_children(child_id)
            
            # 2. Remove child from its layer_map
            child_layer_idx = self._get_layer_index(child_id)
            if child_id in self.layer_map.get(child_layer_idx, {}):
                del self.layer_map[child_layer_idx][child_id]
            
            # 3. Remove from pending_items if it was waiting to be processed
            if child_id in self.pending_items:
                self.pending_items.remove(child_id)

        # Reset the children list of the current item
        item["children"] = []

    def _create_children(self):
        """
        Takes items from self.processing_queue (which now have logits),
        creates children, adds children to layer_map and pending_items.
        """
        while self.processing_queue:
            item_id = self.processing_queue.pop(0)
            layer_index = self._get_layer_index(item_id)
            
            # Check existence in case of aggressive cleanup (edge case)
            if item_id not in self.layer_map.get(layer_index, {}):
                continue

            item = self.layer_map[layer_index][item_id]

            if layer_index >= self.depth:
                item["status"] = "done"
                if "logits" in item:
                    del item["logits"]
                continue
            
            if "logits" not in item:
                logging.error(f"Item {item_id} missing logits during processing.")
                continue
                
            logits = item["logits"]
            
            # --- Top K Logic ---
            probs = mx.softmax(logits, axis=-1)
            top_indices = mx.argpartition(probs, -self.width)[-self.width:]
            
            top_probs_unsorted = probs[top_indices]
            sort_order = mx.argsort(top_probs_unsorted)[::-1]
            final_indices = top_indices[sort_order]
            final_probs = top_probs_unsorted[sort_order]

            # --- Branching ---
            next_layer_index = layer_index + 1
            indices_py = final_indices.tolist()
            probs_py = final_probs.tolist()

            for i in range(len(indices_py)):
                token_id = indices_py[i]
                token_prob = probs_py[i]
                
                token_text = self.tokenizer.decode([token_id])
                new_ids = item["token_ids"] + [token_id]
                
                # Unique ID structure: Layer_ParentID_Index
                new_layer_id = f"{next_layer_index}_{item_id}_{i}"
                
                new_item = {
                    "layer": next_layer_index,
                    "id": new_layer_id,
                    "parent_id": item_id,
                    "token_ids": new_ids,
                    "text": item["text"] + token_text,
                    "probability": token_prob,
                    "last_token_text": token_text,
                    "status": "pending",
                    "children": [],
                }
                
                self.layer_map.setdefault(next_layer_index, {})[new_layer_id] = new_item
                item["children"].append(new_layer_id)
                self.pending_items.append(new_layer_id)

            del item["logits"]
            item["status"] = "done"

    def _run_batch_generation(self):
        """
        Identifies a batch of items at the SAME layer/depth.
        Cleans up their children (in case of re-run).
        Runs the model.
        """
        if not self.pending_items:
            return

        # 1. Find the minimum layer present in pending items
        min_layer = min([self._get_layer_index(pid) for pid in self.pending_items])
        
        # 2. Extract ONLY items at this layer
        batch_ids = []
        remaining_items = []
        
        for pid in self.pending_items:
            if self._get_layer_index(pid) == min_layer:
                batch_ids.append(pid)
            else:
                remaining_items.append(pid)
        
        # Update pending items to exclude the current batch
        self.pending_items = remaining_items
        
        # 3. Prepare Batch & Cleanup Children
        batch_input_ids = []
        
        for item_id in batch_ids:
            # CLEANUP: Before we generate, delete any existing children 
            # to prevent inconsistent tree state
            self._delete_children(item_id)
            
            item = self.layer_map[min_layer][item_id]
            batch_input_ids.append(mx.array(item["token_ids"]))
            self.processing_queue.append(item_id)

        if not batch_input_ids:
            return

        # 4. Run Model
        # All items in this batch are guaranteed to be at min_layer -> same length
        input_tensor = mx.stack(batch_input_ids)

        logging.info(f"Processing Layer {min_layer}: {len(batch_ids)} items")
        
        logits = self.model(input_tensor) 
        next_token_logits = logits[:, -1, :]

        # 5. Assign logits
        for i, item_id in enumerate(batch_ids):
            self.layer_map[min_layer][item_id]["logits"] = next_token_logits[i]

    def start(self):
        while len(self.pending_items) > 0:
            self._run_batch_generation()
            self._create_children()
    
    def to_json(self):
        def default(o):
            if isinstance(o, mx.array):
                return o.tolist()
        return json.dumps(self.layer_map, indent=2, default=default)

def main():
    parser = argparse.ArgumentParser(description="LLM Loom")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=3)
    args = parser.parse_args()

    logging.info(f"Loading model {args.model}...")
    model, tokenizer = load(args.model)
    
    loom = Loom(
        model=model, 
        tokenizer=tokenizer, 
        initial_prompt_text=args.prompt, 
        depth=args.depth, 
        width=args.width
    )
    
    logging.info("Starting Loom...")
    loom.start()
    
    print(loom.to_json())

if __name__ == "__main__":
    main()