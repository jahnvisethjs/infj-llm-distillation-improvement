"""Small CPU tensor tests after software installation. No model weights/GPU."""
import unittest
import torch
import torch.nn.functional as F


class TensorLossChecks(unittest.TestCase):
    def test_causal_shift_prompt_mask_and_eos(self):
        logits=torch.tensor([[2.,0.,1.],[0.,1.,3.],[3.,1.,0.],[0.,0.,0.]],dtype=torch.float64)
        labels=torch.tensor([-100,-100,2,0])
        loss=F.cross_entropy(logits[:-1],labels[1:],ignore_index=-100,reduction='sum')
        manual=-torch.log_softmax(logits,dim=-1)[1,2]-torch.log_softmax(logits,dim=-1)[2,0]
        self.assertTrue(torch.allclose(loss,manual))
        self.assertEqual(int((labels[1:]!=-100).sum()),2)

    def test_pad_equal_eos_keeps_real_eos_supervised(self):
        logits=torch.tensor([[1.,0.,2.],[2.,0.,1.],[3.,1.,0.],[1.,2.,0.]],dtype=torch.float64)
        labels=torch.tensor([-100,2,0,-100,-100])
        padded=F.cross_entropy(logits,labels[1:],ignore_index=-100,reduction='sum')
        unpadded=F.cross_entropy(logits[:2],torch.tensor([2,0]),reduction='sum')
        self.assertTrue(torch.allclose(padded,unpadded))
        self.assertEqual(int((labels[1:]!=-100).sum()),2)

    def test_token_weighted_accumulation_matches_pooled_gradient(self):
        a=torch.tensor([[1.,0.,2.],[2.,0.,1.]],requires_grad=True)
        b=torch.tensor([[0.,1.,2.]],requires_grad=True)
        ya=torch.tensor([2,0]);yb=torch.tensor([1]);total=3
        loss_a=F.cross_entropy(a,ya,reduction='sum')/total
        loss_b=F.cross_entropy(b,yb,reduction='sum')/total
        loss_a.backward();loss_b.backward()
        pooled=torch.cat([a.detach(),b.detach()]).requires_grad_(True)
        F.cross_entropy(pooled,torch.cat([ya,yb]),reduction='mean').backward()
        self.assertTrue(torch.allclose(torch.cat([a.grad,b.grad]),pooled.grad,atol=1e-7))

    def test_chunked_cross_entropy_matches_full_sum(self):
        logits=torch.tensor([[1.,0.,2.],[2.,0.,1.],[1.,2.,0.],[0.,1.,2.],[0.,2.,1.]],requires_grad=True)
        labels=torch.tensor([-100,2,0,1,2])
        full=F.cross_entropy(logits,labels,ignore_index=-100,reduction='sum')
        chunks=sum(F.cross_entropy(logits[i:i+2],labels[i:i+2],ignore_index=-100,reduction='sum') for i in range(0,len(labels),2))
        self.assertTrue(torch.allclose(full,chunks,atol=1e-6))


if __name__=='__main__':unittest.main()
