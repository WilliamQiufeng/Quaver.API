using Quaver.API.Maps.Structures;

namespace Quaver.API.Maps.AutoMod.Issues.HitObjects
{
    public class AutoModIssueOverlappingMine : AutoModIssue
    {
        public override AutoModIssueCategory Category { get; protected set; } =
            AutoModIssueCategory.HitObjects;

        public HitObjectInfo Mine { get; }

        public HitObjectInfo CoveredHitObject { get; }

        public AutoModIssueOverlappingMine(HitObjectInfo mine, HitObjectInfo coveredHitObject) :
            base(AutoModIssueLevel.Ranking)
        {
            Mine = mine;
            CoveredHitObject = coveredHitObject;
            Text =
                $"Mine at {Mine.StartTime} lane {Mine.Lane} should be " +
                $">{AutoMod.OverlappingMineThreshold}ms away from hit object " +
                $"at {CoveredHitObject.StartTime}.";
        }
    }
}